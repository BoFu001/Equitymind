# Tag-Based Similarity Test

Similarity rankings computed against merged LLM-extracted tag
strings (research/industry_tags_comparison.json, llm_tags field),
compared to the v3 full-overview baseline in v3_threshold_baseline.json.

## Query: `technology`

| Rank | Ticker | Score | Tags |
|---|---|---|---|
| 1 | NFLX | 0.5815 | technology, entertainment, media |
| 2 | ADI | 0.5703 | technology, semiconductors, artificial intelligence |
| 3 | GEV | 0.5617 | technology, energy, sustainability |
| 4 | EA | 0.5613 | technology, gaming, entertainment |
| 5 | ANET | 0.5606 | technology, cloud computing, networking |
| 6 | EMR | 0.5602 | technology, industrial manufacturing, artificial intelligence |
| 7 | ITW | 0.5601 | technology, manufacturing, industrial equipment |
| 8 | PLTR | 0.5529 | technology, data analytics, artificial intelligence |
| 9 | QCOM | 0.5481 | technology, semiconductors, wireless communications, artificial intelligence |
| 10 | WDC | 0.5469 | technology, data storage, cloud computing |
| 11 | MCHP | 0.5468 | technology, semiconductors, embedded systems |
| 12 | CMI | 0.5453 | technology, automotive, energy |
| 13 | SNPS | 0.5452 | technology, semiconductor, cloud computing |
| 14 | AMAT | 0.5451 | technology, semiconductors, manufacturing |
| 15 | MDB | 0.5450 | technology, cloud computing, artificial intelligence |

## Query: `financial services`

| Rank | Ticker | Score | Tags |
|---|---|---|---|
| 1 | USB | 0.7800 | financial services, banking, investment management, payment processing |
| 2 | TFC | 0.7773 | financial services, banking, investment, wealth management |
| 3 | MET | 0.7770 | financial services, insurance, asset management, investment |
| 4 | AXP | 0.7750 | financial services, technology, lifestyle services |
| 5 | STT | 0.7537 | financial services, investment management, technology |
| 6 | PNC | 0.7472 | financial services, banking, asset management, corporate finance |
| 7 | NTRS | 0.7441 | financial services, wealth management, asset management, banking |
| 8 | PRU | 0.7431 | financial services, insurance, investment management, retirement solutions |
| 9 | COF | 0.7360 | financial services, banking, credit cards, digital banking |
| 10 | SCHW | 0.7319 | financial services, wealth management, fintech |
| 11 | MS | 0.7279 | financial services, wealth management, investment banking, asset management |
| 12 | GS | 0.7156 | financial services, investment banking, asset management, sustainable finance |
| 13 | JPM | 0.7108 | financial services, investment banking, asset management, commercial banking |
| 14 | SPGI | 0.6883 | financial services, data analytics, energy |
| 15 | CME | 0.6633 | financial services, derivatives trading, market data services, technology |

## Query: `healthcare`

| Rank | Ticker | Score | Tags |
|---|---|---|---|
| 1 | DXCM | 0.6845 | healthcare, medical device |
| 2 | HUM | 0.6644 | healthcare, insurance, managed care |
| 3 | UNH | 0.6415 | health care, insurance, technology |
| 4 | CNC | 0.6397 | healthcare, managed care, insurance |
| 5 | ELV | 0.5918 | healthcare, insurance, pharmacy services, managed care |
| 6 | BSX | 0.5860 | medical technology, healthcare, innovation |
| 7 | KVUE | 0.5843 | healthcare, consumer goods, digital marketing |
| 8 | BDX | 0.5783 | medical technology, healthcare, life sciences |
| 9 | IQV | 0.5758 | healthcare, technology, data analytics, clinical research |
| 10 | VTR | 0.5657 | real estate, healthcare, senior living |
| 11 | WELL | 0.5657 | real estate, healthcare, senior living |
| 12 | CVS | 0.5654 | health care, retail pharmacy, insurance, health services |
| 13 | MDT | 0.5644 | healthcare technology, medical devices, biotechnology |
| 14 | SYK | 0.5606 | medical technology, healthcare, robotics |
| 15 | CI | 0.5592 | healthcare, insurance, pharmacy benefit management |

## Query: `pharmaceuticals`

| Rank | Ticker | Score | Tags |
|---|---|---|---|
| 1 | LLY | 0.7337 | pharmaceuticals, biotechnology, healthcare |
| 2 | JNJ | 0.7226 | pharmaceuticals, medical devices, healthcare |
| 3 | PFE | 0.7065 | pharmaceuticals, biotechnology, healthcare, life sciences |
| 4 | ABT | 0.6840 | pharmaceuticals, diagnostics, nutrition, medical devices |
| 5 | ZTS | 0.6605 | pharmaceuticals, biotechnology, veterinary medicine, diagnostics |
| 6 | MRK | 0.6281 | pharmaceuticals, biotechnology, animal health, oncology |
| 7 | BMY | 0.6226 | biotechnology, pharmaceuticals, healthcare |
| 8 | ABBV | 0.6225 | biotechnology, pharmaceuticals, healthcare |
| 9 | TMO | 0.5880 | technology, healthcare, biotechnology, pharmaceuticals |
| 10 | GILD | 0.5749 | biotechnology, pharmaceuticals, healthcare innovation |
| 11 | REGN | 0.5676 | biotechnology, pharmaceuticals, healthcare, genetic research |
| 12 | VRTX | 0.5444 | biotechnology, pharmaceuticals, gene therapy, specialty medicine |
| 13 | AMGN | 0.5390 | biotechnology, pharmaceuticals, healthcare, oncology |
| 14 | MTD | 0.4889 | technology, manufacturing, life sciences |
| 15 | MDT | 0.4787 | healthcare technology, medical devices, biotechnology |

## Query: `semiconductors`

| Rank | Ticker | Score | Tags |
|---|---|---|---|
| 1 | MPWR | 0.6946 | semiconductor, technology |
| 2 | MCHP | 0.6681 | technology, semiconductors, embedded systems |
| 3 | TXN | 0.6369 | technology, semiconductors, electronics manufacturing |
| 4 | AMAT | 0.6194 | technology, semiconductors, manufacturing |
| 5 | MU | 0.5989 | technology, semiconductors, cloud computing |
| 6 | MRVL | 0.5889 | technology, semiconductors, data infrastructure |
| 7 | ADI | 0.5752 | technology, semiconductors, artificial intelligence |
| 8 | SNPS | 0.5744 | technology, semiconductor, cloud computing |
| 9 | ON | 0.5695 | technology, semiconductors, automotive, industrial automation |
| 10 | LRCX | 0.5651 | technology, semiconductor manufacturing, advanced materials |
| 11 | QCOM | 0.5475 | technology, semiconductors, wireless communications, artificial intelligence |
| 12 | KLAC | 0.5451 | electronics, semiconductor, semiconductor capital equipment, semiconductor manufacturing |
| 13 | AVGO | 0.5451 | technology, semiconductors, cloud computing, cybersecurity |
| 14 | AMD | 0.5273 | technology, semiconductors, ai computing, gaming |
| 15 | COHR | 0.5217 | technology, telecommunications, industrial manufacturing, semiconductor materials |

## Query: `robotics`

| Rank | Ticker | Score | Tags |
|---|---|---|---|
| 1 | SYK | 0.5825 | medical technology, healthcare, robotics |
| 2 | TER | 0.5693 | technology, robotics, semiconductor testing, automation |
| 3 | ISRG | 0.5363 | medical devices, robotics, healthcare innovation |
| 4 | HON | 0.4692 | technology, aerospace, industrial automation, sustainability |
| 5 | EMR | 0.4581 | technology, industrial manufacturing, artificial intelligence |
| 6 | ADI | 0.4310 | technology, semiconductors, artificial intelligence |
| 7 | RTX | 0.4292 | aerospace, defense, technology, engineering |
| 8 | NXPI | 0.4287 | technology, automotive, iot, security |
| 9 | ON | 0.4269 | technology, semiconductors, automotive, industrial automation |
| 10 | LMT | 0.4239 | aerospace, defense, technology, cybersecurity |
| 11 | QCOM | 0.4237 | technology, semiconductors, wireless communications, artificial intelligence |
| 12 | GM | 0.4236 | automotive, technology, electric vehicles |
| 13 | ROK | 0.4206 | industrial automation, digital transformation, technology |
| 14 | NOC | 0.4186 | aerospace, defense, technology |
| 15 | F | 0.4100 | automotive, technology, sustainability |
