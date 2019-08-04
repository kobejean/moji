import XCTest
import TensorFlow
@testable import Moji

final class AttentionTests: XCTestCase {
    
    func testAttentionAttendNoncausal() {
        // Given
        let k = Float(sqrt(sqrt(3.0) * log(2)))
        let query = Tensor<Float>(shape: [1, 3, 3], scalars: [0, 0, 1, 0, 1, 0, 1, 0, 0]) * k
        let key = Tensor<Float>(shape: [1, 3, 3], scalars: [1, 0, 0, 0, 1, 0, 0, 0, 1]) * k
        let value = Tensor<Float>(shape: [1, 3, 3], scalars: (0..<9).map(Float.init))
        // When
        let layer = Attention(size: value.shape[1], causal: false, dropProbability: 0.0)
        // Then
        let output = layer.attend(query: query, key: key, value: value)
        let expected = Tensor<Float>(shape: [3, 3], scalars: [3.75, 4.75, 5.75, 3.0, 4.0, 5.0, 2.25, 3.25, 4.25])
        assertEqual(output, expected, accuracy: 1e-5)
    }
    
    func testAttentionAttendCausal() {
        // Given
        let k = Float(sqrt(sqrt(3.0) * log(2)))
        let query = Tensor<Float>(shape: [1, 3, 3], scalars: [0, 1, 0, 1, 0, 0, 0, 0, 1]) * k
        let key = Tensor<Float>(shape: [1, 3, 3], scalars: [1, 0, 0, 0, 1, 0, 0, 0, 1]) * k
        let value = Tensor<Float>(shape: [1, 3, 3], scalars: (0..<9).map(Float.init))
        // When
        let layer = Attention(size: value.shape[1], causal: true, dropProbability: 0.0)
        // Then
        let output = layer.attend(query: query, key: key, value: value)
        let expected = Tensor<Float>(shape: [3, 3], scalars: [0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.75, 4.75, 5.75])
        assertEqual(output, expected, accuracy: 1e-5)
    }

    static var allTests = [
        ("testAttentionAttendNoncausal", testAttentionAttendNoncausal),
        ("testAttentionAttendCausal", testAttentionAttendCausal),
    ]
}
