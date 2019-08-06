import TensorFlow

struct Embedding: Differentiable {
    var weight: Tensor<Float>
    
    init(weight: Tensor<Float>) {
        self.weight = weight
    }
    
    init(vocabularySize: Int, vectorSize: Int) {
        self.weight = Tensor(randomUniform: [vocabularySize, vectorSize])
    }

    @differentiable(wrt: self)
    func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Float> {
        return weight.gathering(atIndices: input)
    }
}
