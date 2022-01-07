# ALBERT-pytorch-implementation

## developing...
 * 모델의 개념이해를 돕기 위한 구현물로 현재 변수명을 상세히 적었고 변수를 Configuration화 하지 않고 중복하여 적어 구현하였습니다.


### 2022-01-05
 * `MultiHeadAttention.py` 구현 완료


### 2022-01-06
 * `Transformer.py` 구현 완료
   * Position-wise FeedForward Network 구현 
   * Sublayer Connection 구현
   * GELU이용
 * `ALBERT.py` 구현 중 ... (Embedding 미구현)
 
### 2022-01-07
 * `Embeddings.py` 구현 완료
   * Position Embedding 구현
   * Token Embedding 구현
   * Segment Embedding 구현 (SOP 적용할 시 필요하여 일단 구현)
 * `ALBERT.py` 구현 완료
   * Hidden Projection 구현
   * Staked Layer iteration 가능
 


------------------------------------------------





### TODO Lists.
* ~~Embedding 구현~~
* Masked Language Model + SOP 구현
* 변수 Configuration 화
* 코드 정리 및 Package 화

### IDEA (Further Implementations)
* Compression
  * Pruning
  * Quantization (clustering method)
  * Distillation