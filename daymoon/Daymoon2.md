## 1.XGBoost

1. XGBoost란?
2. XGBoost의 특징
3. XGBoost의 파라미터

## 2. 실습
1. 라이브러리 설치
2. 데이터 불러오기
3. EDA
4. train-test split
5. 사이킷런 래퍼 XGBoost 모델링
6. 모델 성능 평가
7. 조기 중단 수행 & 성능 평가
8. Feature Importance 시각화

# 1. XGBoost 
  extra gradient boost: 
  XGBoost는 트리 기반 앙상블 학습( 결합 )에서 가장 각광받고 있는 알고리즘 중 하나. 

  특히 분류에 있어서 일반적으로 다른 머신러닝보다 뛰어난 예측 성능을 보이며, 자체적으로 교차 검증/ 성능 평가 / 변수 중요도 등의 시각화 기능을 가지고 있음

  GBM에 기반한 XGBoost. GBM의 단점인 느린 수행 시간 및 과적합 규제 부재 등의 문제가 해결된 것이다.
  -> 빠른 이유: 병렬 CPU환경에서 병렬 학습을 할 수 있기 때문에
  : 조기 중단 기능도 포함

  기존 GBM의 경우 n_estimators 에 지정된 횟수만큼 학습을 완료하는데, XGBoost는 지정한 부스팅 반복 횟수에 도달하지 않더라도 예측 오류가 더 이상 개선되지 않으면 도중에 학습을 중지

  ### XGBoost의 특징
  [장점]
  * 분류와 회귀 영역에서의 뛰어난 예측 성능
  * GBM 대비 빠른 수행 시간
  * 과적합 규제
  * 결손값 자체 처리
  * 나무 가지치기 : 더 이상 긍정 이득이 없는 분할을 가지치기 해서 분할 수를 더 줄일 수 있다
  * 자체 내장된 교차 검증 : 반복 수행할 때마다 내부적으로 학습 데이터와 평가 데이터에 대한 교차 검증을 실시한다. 지정된 반복 횟수보다 적은 횟수로 교차 검증을 통해 평가 데이터의 평가값이 최적화되면 반복을 중간에 멈출 수 있는 조기 중단 기능도 있다. 


  [구동하는 두가지 방식]
  XGBoost의 핵심 라이브러리는 c/c++로 작성됨
  xgboost 패키지에는 사이킷런과 호환되지 않는
  1. 파이썬 래퍼 XGBoost 모듈과
  사이킷런과 호환되는 
  2. 사이킷런 래퍼 XGBoost 모듈이 있다.

  * 래퍼함수: 실제 함수를 감싼 함수로 실제 함수 호출 시 특별한 동작을 하도록 기능을 덧붙인 함수

  1과 달리 2는 사이킷런의 fit(), predict() 메서드와 같은 사이킷런 고유의 아키텍처와 다양한 유틸리티를 활용 가능

  1은 XGBoost만의 전용 데이터 객체인 DMatrix를 사용. Numpy 또는 Pandas로 되어 있는 학습용, 검증용, 테스트용 데이터를 모두 Dmatrix 객체로 생성하여 모델에 입력해줘야한다. 

  ### code
    dtr=xgb.DMatrix(data=X_tr, label=y_tr)
    dval= xgb.DMatrix(data=X_val, label=y_val)
    dtest= xgb.DMatrix(data=X_test, label=y_test)

  ## XGBoost의 파라미터
    XGBoost는 GBM기반으로 유사한 하이퍼 파라미터를 동일하게 가지고 있으며, 조기 중단, 과적합을 규제하기 위한 하이퍼 파라미터 등이 추가됨

    파이썬 래퍼 XGBoost 1번 과 사이킷런 래퍼 XGBoost 2번 모듈의 일부 하이퍼 파라미터는 동일한 기능을 수행하지만 그 이름이 다르거나 기본값이 다르다. 

  1번 :python 래퍼 XGBoost
  [일반파라미터]
  1. booster : gbtree 또는 gblinear 선택
  2. silent : 출력 메세지를 나타내고 싶지 않을 경우 1로 설정 (기본값은 0)
  3. nthred : CPU의 실행 스레드 개수 조정 ( 기본값은 CPU 전체 스레드 이용)

  [부스터파라미터]
  * eta: GBM의 학습률(learning rate)와 같은 파라미터로 0~1 사이의 값을 지정 (기본값 0.3)<br>
  * num_boost_rounds : GBM의 n_estimators와 같은 파라미터<br>
  * min_child_weight : 과적합 조절을 위해 사용되는 파라미터로 트리에서 추가적으로 가지를 나눌지를 결정하기 위해 필요한 데이터들의 가중치(weight) 총합 이며 값이 클수록 분할을 자제 (기본값 1)
  * gamma : 과적합 조절을 위해 사용되는 파라미터로 트리의 리프 노드를 추가적으로 나눌지를 결정할 최소 손실 감소값으로, 해당값보다 큰 손실(loss)이 감소된 경우 리프노드를 분할. 값이 클수록 과적합 감소 효과 있음 (기본값 0)
  * max_depth : 트리 기반 알고리즘의 max_depth와 같은 파라미터 (기본값 6)
  * sub_sample : GBM의 subsample과 동일한 파라미터 (기본값 1)
  * colsample_bytree : GBM의 max_feature와 유사한 파라미터로, 트리 생성에 필요한 칼럼(변수 즉 feature)를 임의로 샘플링 (기본값 1)
  * lambda : L2 Regularization 적용값으로, 클수록 과적합 감소 효과가 있음 (기본값 1)
  * alpha : L1 Regularization 적용값으로, 클수록 과적합 감소 효과가 있음 (기본값 0)
  * scale_pos_weight : 특정값으로 치우쳐 비대칭하게 구성된 데이터셋의 균형을 유지하기 위한 파라미터 (기본값 1)