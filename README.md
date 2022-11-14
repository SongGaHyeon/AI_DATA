# AI_DATA
📈 AI/DATA 공부 저장소


### 1114~ DAYMOON 참고
* 이해가 안가는 코드가 몇 군데 있었다. 
  1. box plot으로 나타낸 부분에서 코드가 이해가 안갔다. 
  2. GBM 이용 코드에서 random_state를 이용해 난수 생성을 해준 이유를 알고 싶다.
  3. 변수 중요도를 그린 plt에서 

    ftr_importances_values = gb_clf.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns)
    ftr_top15 = ftr_importances.sort_values(ascending=False)[:15]

  이 부분에 대해 이해가 필요하다. 

  4. params 부분에서 작성해준 dictionary(?) 구조가 뭘 위해 쓰여진 코드인지 궁금하다.
  최적의 파라미터를 찾는것이 손실함수의 값을 줄여가는 그 부분인가?

#### 윗 부분 공부를 좀 더 하기 . 오늘의 daymoon_day1 일지 끝. 🤔
