"""DART 공시 원문 구조화 (공급계약 체결·해지·진행사항).

규칙 기반 파서만 쓴다. 텍스트 모델·임베딩·LLM 추출은 쓰지 않는다
(계획: my/milestones/common/research/20261027_investment_research_agenda/07_*.md).
파서는 입출력이 분리된 순수 함수(zip 바이트 → dict)라서, 증분 운영이 정해지면
그대로 ``collector/``로 옮길 수 있다.
"""

PARSER_VERSION = "disclosure_contract_1.0.0"
