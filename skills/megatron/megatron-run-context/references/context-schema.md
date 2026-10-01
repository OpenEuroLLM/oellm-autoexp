# Run context schema

```json
{
  "schema_version": 1,
  "run": {
    "identifier": "run-name",
    "root": "/absolute/run/root",
    "cluster": "local|lumi|tensorwave",
    "scheduler_id": "optional",
    "megatron_revision": "compatibility key or unknown"
  },
  "artifacts": {
    "config": "/absolute/path or null",
    "tensorboard_dir": "/absolute/path or null",
    "log": "/absolute/path or null",
    "availability": {"config": true, "tensorboard_dir": false, "log": true}
  },
  "discovered_at": "ISO-8601 UTC"
}
```
