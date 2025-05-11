<RAPA 인수인계>
-	모든 폴더는 /nas_homes/projects/rapa에 위치.

-	폴더 별 설명
1)	Chatvector
-	Chatvector 모델 위치 (생성 방법 Github 참조)

2)	Dataset 
A.	Blossom (blossom Fine-tuning 용)
B.	Data_updated  = 의료 증강 데이터; PPT 참조 (서울대 + 기존 데이터)
C.	Data_update_512 : 512 이를 512토큰 자름


3)	Results : 이때까지 한 모들 결과
A.	BERT Score, BLEU 등등… -> CoT는 steering vector (https://github.com/Marker-Inc-Korea/COT_steering/tree/main)

4)	Code
A.	Data_checker : 정민님이 주신 데이터 전처리 (위의 data_updated로 이미 완료되어짐)
B.	LoRA_first.py : Openbiollm 모델 학습
Exaone.py : Exaone 모델 학습 코드 (생성한 모델은 용량 문제로 제거, 학습 필요
C.	)
D.	Ds_confing.json : Deepspeed 코드
E.	Generation1.ipynb : 실제 텍스트 생성 코드
F.	나머지 폴더는 권한이 없어 제거하지 못함
