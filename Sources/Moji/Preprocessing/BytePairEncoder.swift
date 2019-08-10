import Python
import TensorFlow

struct BytePairEncoder {
    private let pyencoder: PythonObject
    
    init(modelName: String = "117M") {
        pyencoder = Python.import("moji").encoder.get_encoder(modelName)
    }

    func callAsFunction(_ text: String) -> Tensor<Int32> {
        let pytok = pyencoder.encode(text)
        let tokarr: [Int32] = Array<Int>(pytok)!.map { Int32($0) }
        return Tensor(shape: [1, tokarr.count], scalars: tokarr)
    }
}
