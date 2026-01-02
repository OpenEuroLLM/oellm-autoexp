# Escaped Interpolation Design

## Problem

Multi-stage sweeps need interpolations (like `${sibling.stable.output_dir}`) that reference values not available during config loading. These should only resolve later during DAG resolution.

## Challenge

OmegaConf automatically unescapes `\\$` → `$` during `to_container(resolve=True)`, making escaped interpolations "live" before their referenced values exist.

## Solution: Generic Escaped Dollar Replacement

Instead of pattern-matching specific prefixes (sibling, aux), we use a **fully generic approach**:

### Algorithm

1. Load config with Hydra (get OmegaConf object)
2. Get **unresolved** container: `to_container(resolve=False)`
3. **Generic replacement**: ALL `\\$` → `__ESCAPED_DOLLAR__` (works for ANY interpolation)
4. Create new OmegaConf from modified container
5. Resolve: `to_container(resolve=True)` (escaped interpolations stay as literals)
6. During sweep expansion: keep escaped
7. During DAG resolution: Replace back `__ESCAPED_DOLLAR__` → `$`, then resolve

### Benefits

- **Extensible**: Adding `\\${slurm.sbatch.nodes}` in config works automatically
- **No pattern matching**: Don't need to know what's being escaped
- **No hardcoded paths**: Works for any `\\${...}` pattern
- **Minimal code**: Single string replacement, not regex patterns

### YAML Usage

In configs, escape ANY interpolation that should defer resolution:

```yaml
# Standard interpolation (resolves during config loading)
backend.megatron.lr: ${base_lr}

# Escaped interpolation (resolves during DAG resolution)
backend.megatron.load: "\\${sibling.stable.output_dir}/checkpoints"
slurm.sbatch.nodes: "\\${oc.eval:...}"  # Also works!
any.nested.field: "\\${custom.reference}"  # Also works!
```

### Implementation

Only need to handle:
- `{{...}}` placeholders (for Python str.format, not OmegaConf)
- `\\$` → generic placeholder → `$` (for deferred OmegaConf interpolations)

No knowledge of sibling, aux, or any specific patterns required!

### Note on Escaping in OmegaConf Interpolations

According to [OmegaConf grammar documentation](https://omegaconf.readthedocs.io/en/latest/grammar.html), **parentheses must be escaped** in unquoted strings within interpolations.

In YAML, you need to double-escape (due to YAML parsing):
- `\\$` → becomes `\$` after YAML parsing (escapes dollar sign)
- `\\(` → becomes `\(` after YAML parsing (escapes left paren)
- `\\)` → becomes `\)` after YAML parsing (escapes right paren)

```yaml
# CORRECT - parentheses are escaped
train_iters: "\\${oc.eval:\\(\\${backend.megatron.aux.tokens}+1\\)}"
# After YAML parsing, OmegaConf sees: \${oc.eval:\(\${backend.megatron.aux.tokens}+1\)}

# WRONG - would cause OmegaConf parsing errors
train_iters: "\\${oc.eval:(\\${backend.megatron.aux.tokens}+1)}"
```

**Why this works:**
1. YAML parser: `\\$` → `\$`, `\\(` → `\(`, `\\)` → `\)`
2. Our loader: `\$` → `__ESCAPED_DOLLAR__` (defers interpolation), `\(` and `\)` stay unchanged
3. Sweep expansion: placeholders preserved as literals
4. DAG resolution: `__ESCAPED_DOLLAR__` → `$`, then OmegaConf resolves with `\(` and `\)` as escaped parens

## Example: Verified Working Configuration

The `dense_300M_50BT_pull.yaml` config demonstrates this approach with different checkpoint iterations per stage:

- **decay6B**: iter_18000
- **decay12B**: iter_36000
- **decay30B**: iter_90000
- **decay50B**: iter_152000

Each stage correctly resolves `\\${backend.megatron.aux.start_iter_round}` with its own token value, without any hardcoded field names in the Python code.
