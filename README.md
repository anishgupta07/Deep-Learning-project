| # | Endpoint | Method | Input | Purpose / What It Returns |
| :---: | --- | :---: | --- | --- |
| 1 | `/healthz` | `GET` | None | Health Check: Returns `{"status": "healthy"}` and confirms the PyTorch model is ready in RAM. |
| 2 | `/v1/predict/variant` | `POST` | JSON (gene, pos, ref, alt) | Single Variant Scoring: Returns ACMG classification (`Pathogenic`, `VUS`, etc.), calibrated PPP, and lifetime cancer risk. |
| 3 | `/v1/predict/vcf` | `POST` | `.vcf` file upload | Batch Patient Analysis: Evaluates entire exome/panel VCF, filters sequencing noise, and aggregates organ cancer risks. |
| 4 | `/docs` | `GET` | None | Interactive Swagger UI: Web documentation to test all endpoints live from the browser. |
