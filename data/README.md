# data/

원본 데이터는 용량(사용하는 3개 파일 합계 약 7.7GB, 노션 표 기준)이 커서 저장소에 포함하지 않는다.

## 1. 원본 받기 (`data/raw/`)

Kaggle: [MIT-Stanford Dataset — data-driven-prediction-of-battery-cycle](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)

| 파일 | Batch | 활용 |
|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` (2.8GB) | Batch 1 | 학습 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` (1.9GB) | Batch 2 | 테스트 |
| `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` (0.1GB) | extra | 사용 안 함 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` (3.0GB) | Batch 3 | 추가 검증 |

```bash
pip install kagglehub==1.0.2      # requirements.txt 와 같은 버전 (pip install -r requirements.txt 로도 설치됨)
python -c "import kagglehub; [kagglehub.dataset_download('itshpark/data-driven-prediction-of-battery-cycle', path=f, output_dir='data/raw') for f in ['2017-05-12_batchdata_updated_struct_errorcorrect.mat','2018-02-20_batchdata_updated_struct_errorcorrect.mat','2018-04-12_batchdata_updated_struct_errorcorrect.mat']]"
```

## 2. 분석용 추출 (`data/processed/`)

```bash
python src/preprocess.py        # → data/processed/batch1.pkl, batch2.pkl, batch3.pkl
python -c "import sys; sys.path.insert(0,'src'); import data as D; D.cell_table(D.load_all()).to_csv('data/processed/cell_table.csv', index=False)"   # 셀 목록·라벨 요약 (참고용)
```

`.mat`(MATLAB v7.3/HDF5)에서 필요한 필드(summary, Qdlin/Tdlin/dQdV 첫 201 사이클, 일부 사이클 원시 시계열)만 읽어 셀 단위로 저장한다.
데이터 규칙(사이클 번호, 중도절단 셀, 측정 오류 처리)은 [`docs/data_notes.md`](../docs/data_notes.md) 참고.
