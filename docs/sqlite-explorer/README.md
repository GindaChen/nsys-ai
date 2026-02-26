# SQLite Schema Explorer

Interactive HTML documentation for the Nsight SQLite export schema.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Main viewer — CSS + layout + tabs |
| `sqlite-explorer-data.js` | All data (tables, concepts, examples, FK relationships) + rendering logic |

## Living Documentation Standard

This explorer is the **single source of truth** for the Nsight SQLite schema. When writing new analysis code that uses SQL:

1. Add your query to `EXAMPLES` in `sqlite-explorer-data.js`
2. Add any new tables/columns to `TABLES`
3. Add FK relationships to `FK_RELATIONSHIPS`
4. Test by opening `index.html` in a browser

## Future Modules

As the explorer grows, split into:
- `tables.js` — table definitions
- `examples.js` — query examples
- `relationships.js` — FK data
- `concepts.js` — key concepts
- `render.js` — rendering logic

## Live

Deployed at `/nsight/` on the production frontend.
