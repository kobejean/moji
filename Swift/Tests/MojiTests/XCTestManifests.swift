import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        // Layers
        // Layers/Attention
        testCase(AttentionTests.allTests),
    ]
}
#endif
