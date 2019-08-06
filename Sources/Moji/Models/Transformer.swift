import TensorFlow

public struct Transformer {
    
    // MARK: Feed Forward
    
    struct FeedForward: Layer {
        var dense1: TimeDistributed
        var dense2: TimeDistributed
        @noDerivative let dropout: Dropout<Float>

        init(size: Int, hidden: Int, dropProbability: Double) {
            dense1 = TimeDistributed(Dense<Float>(inputSize: size, outputSize: hidden, activation: gelu))
            dense2 = TimeDistributed(Dense<Float>(inputSize: hidden, outputSize: size))
            dropout = Dropout<Float>(probability: dropProbability)
        }

        @differentiable(wrt: (self, input))
        func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            return input.sequenced(through: dense1, dropout, dense2)
        }
    }
    
    // MARK: Encoder
    
    public struct Encoder: Layer {
        var selfAttention: MultiHeadAttention
        @noDerivative let selfAttentionDropout: Dropout<Float>
        var selfAttentionNorm: LayerNorm<Float>
        var feedForward: FeedForward
        @noDerivative let feedForwardDropout: Dropout<Float>
        var feedForwardNorm: LayerNorm<Float>
        
        public init(size: Int, headCount: Int, dropProbability: Double) {
            self.selfAttention = MultiHeadAttention(size: size, headCount: headCount)
            self.selfAttentionDropout = Dropout(probability: dropProbability)
            self.selfAttentionNorm = LayerNorm(featureCount: size, axis: 2, epsilon: Tensor<Float>(1e-5))
            self.feedForward = FeedForward(size: size, hidden: 4 * size, dropProbability: dropProbability)
            self.feedForwardDropout = Dropout(probability: dropProbability)
            self.feedForwardNorm = LayerNorm(featureCount: size, axis: 2, epsilon: Tensor<Float>(1e-5))
        }
        
        @differentiable(wrt: (self, input))
        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            let attended = input + input.sequenced(through: selfAttentionNorm, selfAttention, selfAttentionDropout)
            return attended + attended.sequenced(through: feedForwardNorm, feedForward, feedForwardDropout)
        }
        
        @differentiable(wrt: (self, input))
        public func callAsFunction(_ input: Tensor<Float>, state: inout AttentionContext) -> Tensor<Float> {
            var residual = input
            // Self-Attention
            var attended = selfAttentionNorm(residual)
            attended = selfAttention(attended, state: &state)
            attended = selfAttentionDropout(attended)
            
            residual = residual + attended
            // Feed Forward
            var inferred = feedForwardNorm(residual)
            inferred = feedForward(inferred)
            inferred = feedForwardDropout(inferred)
            
            return residual + inferred
        }
    }
    
}
