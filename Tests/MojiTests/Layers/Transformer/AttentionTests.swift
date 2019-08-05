import XCTest
import TensorFlow
@testable import Moji

final class AttentionTests: XCTestCase {
    
    // MARK: Attend Tests
    
    func testAttentionAttendNoncausal() {
        // Given
        let k = Float(sqrt(sqrt(3.0) * log(2)))
        let query = Tensor<Float>(shape: [1, 3, 3], scalars: [0, 0, 1, 0, 1, 0, 1, 0, 0]) * k
        let key = Tensor<Float>(shape: [1, 3, 3], scalars: [1, 0, 0, 0, 1, 0, 0, 0, 1]) * k
        let value = Tensor<Float>(shape: [1, 3, 3], scalars: (0..<9).map(Float.init))
        let layer = Attention(size: value.shape[1], causal: false, dropProbability: 0.0)
        // When
        let output = layer.attend(query: query, key: key, value: value)
        // Then
        let expected = Tensor<Float>(shape: [1, 3, 3], scalars: [3.75, 4.75, 5.75, 3.0, 4.0, 5.0, 2.25, 3.25, 4.25])
        assertEqual(output, expected, accuracy: 1e-5)
    }
    
    func testAttentionAttendNoncausalPullback() {
        // Given
        let k = Float(sqrt(sqrt(3.0) * log(2)))
        let query = Tensor<Float>(shape: [1, 3, 3], scalars: [0, 0, 1, 0, 1, 0, 1, 0, 0]) * k
        let key = Tensor<Float>(shape: [1, 3, 3], scalars: [1, 0, 0, 0, 1, 0, 0, 0, 1]) * k
        let value = Tensor<Float>(shape: [1, 3, 3], scalars: (0..<9).map(Float.init))
        let input = AttentionQueryKeyValue(query: query, key: key, value: value)
        let layer = Attention(size: value.shape[1], causal: false, dropProbability: 0.0)
        let identityMatrix = Tensor<Float>(shape: [1, 3, 3], scalars: [1, 0, 0, 0, 1, 0, 0, 0, 1])
        // When
        let output = pullback(at: input) { input in
            layer.attend(query: input.query, key: input.key, value: input.value)
        } (identityMatrix)
        // Then
        let expected = AttentionQueryKeyValue.AllDifferentiableVariables(
            query: Tensor<Float>(shape: [1, 3, 3], scalars:  [
                 -0.5930669, -0.11861337,  0.71168035,
                -0.47445348,         0.0,  0.47445348,
                -0.71168035,  0.11861337,   0.5930669
            ]),
            key: Tensor<Float>(shape: [1, 3, 3], scalars: [
                -0.71168035, -0.47445348,  -0.5930669,
                 0.11861337,         0.0, -0.11861337,
                  0.5930669,  0.47445348,  0.71168035
            ]),
            value: Tensor<Float>(shape: [1, 3, 3], scalars: [
                 0.24999999,  0.24999999,         0.5,
                 0.24999999,         0.5,  0.24999999,
                        0.5,  0.24999999,  0.24999999
            ])
        )
        
        assertEqual(output.query, expected.query, accuracy: 1e-5, "query gradient is not equal")
        assertEqual(output.key, expected.key, accuracy: 1e-5, "key gradient is not equal")
        assertEqual(output.value, expected.value, accuracy: 1e-5, "value gradient is not equal")
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
        let expected = Tensor<Float>(shape: [1, 3, 3], scalars: [0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.75, 4.75, 5.75])
        assertEqual(output, expected, accuracy: 1e-5)
    }
    
    func testAttentionAttendCausalPullback() {
        // Given
        let k = Float(sqrt(sqrt(3.0) * log(2)))
        let query = Tensor<Float>(shape: [1, 3, 3], scalars: [0, 1, 0, 1, 0, 0, 0, 0, 1]) * k
        let key = Tensor<Float>(shape: [1, 3, 3], scalars: [1, 0, 0, 0, 1, 0, 0, 0, 1]) * k
        let value = Tensor<Float>(shape: [1, 3, 3], scalars: (0..<9).map(Float.init))
        let input = AttentionQueryKeyValue(query: query, key: key, value: value)
        let layer = Attention(size: value.shape[1], causal: true, dropProbability: 0.0)
        let identityMatrix = Tensor<Float>(shape: [1, 3, 3], scalars: [1, 0, 0, 0, 1, 0, 0, 0, 1])
        // When
        let output = pullback(at: input) { input in
            layer.attend(query: input.query, key: input.key, value: input.value)
            } (identityMatrix)
        // Then
        let expected = AttentionQueryKeyValue.AllDifferentiableVariables(
            query: Tensor<Float>(shape: [1, 3, 3], scalars:  [
                        0.0,         0.0,         0.0,
                 -0.4217365,  0.42173645,         0.0,
                 -0.5930669, -0.11861337,  0.71168035
            ]),
            key: Tensor<Float>(shape: [1, 3, 3], scalars: [
                 -0.4217365,         0.0,  -0.5930669,
                 0.42173645,         0.0, -0.11861337,
                        0.0,         0.0,  0.71168035
            ]),
            value: Tensor<Float>(shape: [1, 3, 3], scalars: [
                        1.0,   0.6666667,  0.24999999,
                        0.0,   0.3333333,  0.24999999,
                        0.0,         0.0,         0.5
            ])
        )
        
        assertEqual(output.query, expected.query, accuracy: 1e-5, "query gradient is not equal")
        assertEqual(output.key, expected.key, accuracy: 1e-5, "key gradient is not equal")
        assertEqual(output.value, expected.value, accuracy: 1e-5, "value gradient is not equal")
    }

    
    // MARK: Call As Function Tests
    
    func testAttentionCallAsFunctionNoncausal() {
        // Given
        let k = Float(sqrt(sqrt(3.0) * log(2)))
        let input = Tensor<Float>(shape: [1, 3, 9], scalars: [
            0, 0, k, k, 0, 0, 0, 1, 2,
            0, k, 0, 0, k, 0, 3, 4, 5,
            k, 0, 0, 0, 0, k, 6, 7, 8,
        ])
        let layer = Attention(size: input.shape[1], causal: false, dropProbability: 0.0)
        // When
        let output = layer(input)
        // Then
        let expected = Tensor<Float>(shape: [1, 3, 3], scalars: [3.75, 4.75, 5.75, 3.0, 4.0, 5.0, 2.25, 3.25, 4.25])
        assertEqual(output, expected, accuracy: 1e-5)
    }
    
//    func testAttentionCallAsFunctionCausal() {
//        // Given
//        let k = Float(sqrt(sqrt(3.0) * log(2)))
//        let input = Tensor<Float>(shape: [1, 3, 9], scalars: [
//            0, k, 0, k, 0, 0, 0, 1, 2,
//            k, 0, 0, 0, k, 0, 3, 4, 5,
//            0, 0, k, 0, 0, k, 6, 7, 8,
//        ])
//        let layer = Attention(size: input.shape[1], causal: true, dropProbability: 0.0)
//        // When
//        let output = layer(input)
//        // Then
//        let expected = Tensor<Float>(shape: [1, 3, 3], scalars: [0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.75, 4.75, 5.75])
//        assertEqual(output, expected, accuracy: 1e-5)
//    }
    
    func testAttentionCallAsFunctionCausalPullback() {
        // Given
        let k = Float(sqrt(sqrt(3.0) * log(2)))
        let input = Tensor<Float>(shape: [1, 3, 9], scalars: [
            0, k, 0, k, 0, 0, 0, 1, 2,
            k, 0, 0, 0, k, 0, 3, 4, 5,
            0, 0, k, 0, 0, k, 6, 7, 8,
        ])
        let layer = Attention(size: input.shape[1], causal: true, dropProbability: 0.0)
        let identityMatrix = Tensor<Float>(shape: [1, 3, 3], scalars: [1, 0, 0, 0, 1, 0, 0, 0, 1])
        // When
        let output = pullback(at: input) { input in
            layer(input)
        } (identityMatrix)
        // Then
        let expected = Tensor<Float>.AllDifferentiableVariables([
                    0.0,         0.0,        0.0, -0.4217365, 0.0,  -0.5930669, 1.0, 0.6666667, 0.25,
             -0.4217365,  0.42173645,        0.0, 0.42173645, 0.0, -0.11861337, 0.0, 0.3333333, 0.25,
             -0.5930669, -0.11861337, 0.71168035,        0.0, 0.0,  0.71168035, 0.0,       0.0,  0.5
        ])
        assertEqual(output.scalars, expected.scalars, accuracy: 1e-5)
    }
    
    static var allTests = [
        // Attend Tests
        ("testAttentionAttendNoncausal", testAttentionAttendNoncausal),
        ("testAttentionAttendNoncausalPullback", testAttentionAttendNoncausalPullback),
        ("testAttentionAttendCausal", testAttentionAttendCausal),
        ("testAttentionAttendCausalPullback", testAttentionAttendCausalPullback),
        // Call As Function Tests
        ("testAttentionCallAsFunctionNoncausal", testAttentionCallAsFunctionNoncausal),
//        ("testAttentionCallAsFunctionCausal", testAttentionCallAsFunctionCausal),
        ("testAttentionCallAsFunctionCausalPullback", testAttentionCallAsFunctionCausalPullback),
    ]
}
