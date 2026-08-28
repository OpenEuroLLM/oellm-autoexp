# TensorBoard scalar artifact

```json
{
  "schema_version": 1,
  "event_files": ["/absolute/events.out.tfevents..."],
  "scalars": {
    "moe/example": [{"step": 100, "wall_time": 0.0, "value": 0.5, "source": "/absolute/event"}]
  },
  "missing_tags": ["requested/but/absent"],
  "available_tags": ["all/observed/tags"]
}
```

Values are raw event records. Consumers decide windows and aggregation; do not replace absent tags with
zeros or averages.
