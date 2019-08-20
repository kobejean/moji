import TensorFlow


// MARK: Transformer

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
        
        public init(size: Int, headCount: Int, dropProbability: Double = 0.2) {
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
        public func callAsFunction(_ input: Tensor<Float>, state: inout AttentionContext, mask: Tensor<Float>? = nil) -> Tensor<Float> {
            var residual = input
            // Self-Attention
            var attended = selfAttentionNorm(residual)
            attended = selfAttention(attended, state: &state, mask: mask)
            attended = selfAttentionDropout(attended)
            
            residual = residual + attended
            // Feed Forward
            var inferred = feedForwardNorm(residual)
            inferred = feedForward(inferred)
            inferred = feedForwardDropout(inferred)
            
            return residual + inferred
        }
    }
    
    // MARK: HyperParameters
    
    struct HyperParameters {
        var vocabSize = 50257
        var contextSize = 512
        var embeddingSize = 64 * 12
        var headCount = 12
        var layerCount = 3
    }
    
    
    // MARK: Properties
    let hyperParams: HyperParameters
    let bytePairEncoder: BytePairEncoder
    var embedding: Embedding
    var positionalEmbeddings: Tensor<Float>
    var layers: [Encoder] = []
    var norm: LayerNorm<Float>
    var state: [AttentionContext] = []
    
    // MARK: Initializer
    
    init(hyperParams: HyperParameters = HyperParameters()) {
        self.hyperParams = hyperParams
        self.bytePairEncoder = BytePairEncoder()
        self.embedding = Embedding(vocabSize: hyperParams.vocabSize, featureSize: hyperParams.embeddingSize)
        self.positionalEmbeddings = Tensor(randomUniform: [hyperParams.contextSize, hyperParams.embeddingSize])
        for _ in 0..<hyperParams.layerCount {
            self.layers.append(Encoder(size: hyperParams.embeddingSize, headCount: hyperParams.headCount))
        }
        self.norm = LayerNorm(featureCount: hyperParams.embeddingSize, axis: -1, epsilon: Tensor(1e-5))
        self.resetState()
    }

    
    // MARK: Methods
    
    mutating func callAsFunction(_ text: String) -> Tensor<Float> {
        let tokens = bytePairEncoder.callAsFunction(text)
        let positions = (0..<tokens.shape[1]).map { Int32($0 + state[0].key.shape[1]) }
        let positionsTensor = Tensor<Int32>(shape: [1, tokens.shape[1]], scalars: positions)
        var h = embedding(tokens) + positionalEmbeddings.gathering(atIndices: positionsTensor)
        for i in 0..<layers.count {
            // Remove the .call when TF-516 is fixed.
            h = layers[i].callAsFunction(h, state: &state[i])
        }
        h = norm(h)
        let tmp = TimeDistributed(Dense(weight: embedding.weight.transposed(), bias: Tensor(0.0), activation: identity))
        let logits = tmp(h) // a somewhat hacky way to share weights
        return logits
    }

    
    // MARK: Private Methods
    
    private mutating func resetState() {
        let empty = Tensor<Float>(zeros: [hyperParams.headCount, 0, hyperParams.embeddingSize / hyperParams.headCount])
        self.state = []
        for _ in 0..<layers.count {
            self.state.append(AttentionContext(key: empty, value: empty))
        }
    }
    
}
