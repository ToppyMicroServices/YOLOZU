import AppKit
import PDFKit
import Foundation

let input = URL(fileURLWithPath: CommandLine.arguments[1])
let destination = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
guard let document = PDFDocument(url: input) else { fatalError("Cannot read PDF") }
try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
for index in 0..<document.pageCount {
    try autoreleasepool {
        guard let page = document.page(at: index) else { fatalError("Missing page") }
        let size = page.bounds(for: .mediaBox).size
        let thumbnail = page.thumbnail(of: size, for: .mediaBox)
        guard let tiff = thumbnail.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let png = bitmap.representation(using: .png, properties: [:]) else {
            fatalError("Cannot render page")
        }
        try png.write(to: destination.appendingPathComponent(String(format: "page-%03d.png", index + 1)))
    }
}
let bundle = Bundle(for: PDFDocument.self)
print("PDFKit \(bundle.infoDictionary?["CFBundleShortVersionString"] ?? "unknown") / \(ProcessInfo.processInfo.operatingSystemVersionString); pages=\(document.pageCount)")
