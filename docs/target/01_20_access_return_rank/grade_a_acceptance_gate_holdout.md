# Grade A candidates — Phase 2 확정 holdout 결과 (1회 한정)

> holdout: 2026-06-11 ~ 2026-07-31 (36 거래일) — 이 모델에서 어떤 결정에도 쓰인 적 없는 신규 구간
> git commit at run time: `8e00e8277fc99bdc18fd1ef69733164209210ee7`

| config | horizon | holdout Rank IC | ICIR | top-decile spread | cost-adj spread(60bp 왕복 가정) |
|---|---|---|---|---|---|
| baseline | 5d | **0.2097** | 2.958 | 0.0106 | 0.0169 |
| baseline | 20d | **0.2431** | 4.585 | 0.0298 | 0.0859 |
| baseline | 60d | **nan** | nan | nan | nan |
| candidate | 5d | **0.1912** | 2.864 | 0.0104 | 0.0148 |
| candidate | 20d | **0.3347** | 11.254 | 0.0307 | 0.0182 |
| candidate | 60d | **nan** | nan | nan | nan |

> 이 결과 파일이 존재하는 한 재실행하지 않는다(`--force` 없이는 거부) — 동일 holdout을 두 번 열어보면 다음 결정에 선택 편향이 생긴다.
