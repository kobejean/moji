import XCTest
import TensorFlow

internal func assertEqual<T: TensorFlowFloatingPoint>(
    _ x: [T], _ y: [T], accuracy: T, _ message: String = "",
    file: StaticString = #file, line: UInt = #line
) {
    for (x, y) in zip(x, y) {
        if x.isNaN || y.isNaN {
            XCTAssertTrue(x.isNaN && y.isNaN, "scalar \(x) is not equal to \(y) - \(message)", file: file, line: line)
            continue
        }
        XCTAssertEqual(x, y, accuracy: accuracy, message, file: file, line: line)
    }
}

internal func assertEqual<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>, _ y: Tensor<T>, accuracy: T, _ message: String = "",
    file: StaticString = #file, line: UInt = #line
) {
    assertEqual(x.shape, y.shape, "shape \(x.shape) is not equal to \(y.shape) - \(message)", file: file, line: line)
    assertEqual(x.scalars, y.scalars, accuracy: accuracy, message, file: file, line: line)
}

internal func assertEqual(
    _ x: TensorShape, _ y: TensorShape, _ message: String = "",
    file: StaticString = #file, line: UInt = #line
) {
    for (x, y) in zip(x.dimensions, y.dimensions) {
        XCTAssertEqual(x, y, message, file: file, line: line)
    }
}
