import signal
import subprocess
import sys
import threading
from collections.abc import Sequence

try:  # pragma: no cover - pty only available on POSIX
    import os
    import pty
except ImportError:  # pragma: no cover
    os = None  # type: ignore
    pty = None  # type: ignore


def run_with_tee(
    args: str | Sequence[str],
    *,
    capture_output: bool = True,
    print_cmd: bool = True,
    check: bool = False,
    timeout: float | None = None,
    input: str | bytes | None = None,
    text: bool | None = None,  # None = follow Python default; True = text mode; False = bytes
    encoding: str | None = None,
    errors: str | None = None,
    env=None,
    cwd=None,
    shell: bool = False,
    use_pty: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command like subprocess.run but:

      - streams stdout/stderr live to this process' stdout/stderr (tee)
      - returns a CompletedProcess with captured stdout/stderr
      - supports check, timeout, input, text/encoding/errors

    NOTE: We always pipe child stdout/stderr to implement tee behavior.
    """

    print("RUNNING: ", " ".join(args))

    master_fd: int | None = None
    slave_fd: int | None = None

    popen_kwargs = {
        "stdin": subprocess.PIPE if input is not None else None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": env,
        "cwd": cwd,
        "shell": shell,
        "text": text,
        "encoding": encoding,
        "errors": errors,
    }

    if use_pty:
        if pty is None or os is None:  # pragma: no cover - non POSIX fallback
            raise RuntimeError("PTY mode is only supported on POSIX systems")
        master_fd, slave_fd = pty.openpty()
        popen_kwargs["stdout"] = slave_fd
        popen_kwargs["stderr"] = slave_fd
        popen_kwargs["bufsize"] = 0

    proc = subprocess.Popen(args, **popen_kwargs)

    if use_pty and slave_fd is not None:
        os.close(slave_fd)

    # Select console targets and buffers depending on text/binary mode
    is_text = proc.stdout is not None and (text or encoding or errors) is not None or (text is True)
    if is_text is None:  # follow Popen's actual mode
        is_text = proc.text if hasattr(proc, "text") else False

    out_buf = []
    err_buf = []

    def _forward(src, sink, buf, chunk_bytes: int = 8192):
        if src is None:
            return
        if is_text:
            for line in iter(src.readline, ""):
                buf.append(line)
                sink.write(line)
                sink.flush()
        else:
            bsink = sink.buffer if hasattr(sink, "buffer") else sink
            for chunk in iter(lambda: src.read(chunk_bytes), b""):
                buf.append(chunk)
                bsink.write(chunk)
                bsink.flush()
        src.close()

    def _forward_pty(master: int, sink, buf, chunk_bytes: int = 1024):
        if os is None:
            return
        try:
            while True:
                try:
                    chunk = os.read(master, chunk_bytes)
                except OSError:
                    break
                if not chunk:
                    break
                if is_text:
                    text_chunk = chunk.decode(encoding or "utf-8", errors or "replace")
                    buf.append(text_chunk)
                    sink.write(text_chunk)
                    sink.flush()
                else:
                    buf.append(chunk)
                    bsink = sink.buffer if hasattr(sink, "buffer") else sink
                    bsink.write(chunk)
                    bsink.flush()
        finally:
            os.close(master)

    if use_pty and master_fd is not None:
        t_out = threading.Thread(target=_forward_pty, args=(master_fd, sys.stdout, out_buf))
        t_err = None
    else:
        t_out = threading.Thread(target=_forward, args=(proc.stdout, sys.stdout, out_buf))
        t_err = threading.Thread(target=_forward, args=(proc.stderr, sys.stderr, err_buf))

    t_out.start()
    if t_err is not None:
        t_err.start()

    # Write input (if any)
    if input is not None:
        try:
            if is_text or isinstance(input, str):
                data = (
                    input
                    if isinstance(input, str)
                    else input.decode(encoding or "utf-8", errors or "strict")
                )
            else:
                data = input if isinstance(input, (bytes, bytearray)) else str(input).encode()
            proc.stdin.write(data)  # type: ignore[union-attr]
        finally:
            proc.stdin.close()  # type: ignore[union-attr]

    try:
        retcode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # Kill, finish draining, then raise with partial output
        proc.kill()
        proc.wait()
        t_out.join()
        if t_err is not None:
            t_err.join()
        captured_stdout = "".join(out_buf) if is_text else b"".join(out_buf)
        captured_stderr = "".join(err_buf) if is_text else b"".join(err_buf)
        e.output = captured_stdout
        e.stderr = captured_stderr
        raise
    except KeyboardInterrupt:
        # Propagate SIGINT to child process so it can clean up.
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

        try:
            retcode = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # escalate to SIGTERM
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                retcode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                retcode = proc.wait()

        t_out.join()
        if t_err is not None:
            t_err.join()

        captured_stdout = "".join(out_buf) if is_text else b"".join(out_buf)
        captured_stderr = "".join(err_buf) if is_text else b"".join(err_buf)

        raise KeyboardInterrupt from None

    t_out.join()
    if t_err is not None:
        t_err.join()

    captured_stdout = "".join(out_buf) if is_text else b"".join(out_buf)
    captured_stderr = "".join(err_buf) if is_text else b"".join(err_buf)

    if check and retcode != 0:
        raise subprocess.CalledProcessError(
            retcode, args, output=captured_stdout, stderr=captured_stderr
        )

    return subprocess.CompletedProcess(
        args=args,
        returncode=retcode,
        stdout=captured_stdout,
        stderr=captured_stderr,
    )
