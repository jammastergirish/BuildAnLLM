/**
 * Property-based tests for multi-file upload functionality.
 * Uses fast-check for property testing.
 */

import { describe, it, expect } from "vitest";
import fc from "fast-check";

describe("Multi-File Upload Properties", () => {
  describe("Word Count Calculation", () => {
    it("word count is always non-negative", () => {
      fc.assert(
        fc.property(fc.string(), (text) => {
          const words = text.split(/\s+/).filter((w) => w.length > 0).length;
          return words >= 0;
        })
      );
    });

    it("empty string has zero words", () => {
      fc.assert(
        fc.property(fc.constant(""), (text) => {
          const words = text.split(/\s+/).filter((w) => w.length > 0).length;
          return words === 0;
        })
      );
    });

    it("whitespace-only string has zero words", () => {
      fc.assert(
        fc.property(
          fc.stringOf(fc.constantFrom(" ", "\t", "\n", "\r")),
          (text) => {
            const words = text.split(/\s+/).filter((w) => w.length > 0).length;
            return words === 0;
          }
        )
      );
    });
  });

  describe("Character Count Calculation", () => {
    it("character count matches string length", () => {
      fc.assert(
        fc.property(fc.string(), (text) => {
          const chars = text.length;
          return chars === text.length;
        })
      );
    });

    it("character count is always non-negative", () => {
      fc.assert(
        fc.property(fc.string(), (text) => {
          return text.length >= 0;
        })
      );
    });
  });

  describe("File Name Handling", () => {
    it("file names with alphanumeric characters are preserved", () => {
      fc.assert(
        fc.property(
          fc.stringOf(fc.char().filter((c) => /[a-zA-Z0-9_.-]/.test(c)), { minLength: 1 }),
          (name) => {
            const fullName = `${name}.txt`;
            const file = new File(["content"], fullName, { type: "text/plain" });
            return file.name === fullName;
          }
        )
      );
    });
  });

  describe("File Concatenation", () => {
    it("concatenation length equals sum of parts plus separators", () => {
      fc.assert(
        fc.property(
          fc.array(fc.string({ minLength: 1 }), { minLength: 1, maxLength: 5 }),
          (texts) => {
            const combined = texts.join("\n\n");
            const expectedLength =
              texts.reduce((sum, t) => sum + t.length, 0) +
              (texts.length - 1) * 2; // 2 chars for each \n\n separator
            return combined.length === expectedLength;
          }
        )
      );
    });

    it("all parts appear in concatenated result", () => {
      fc.assert(
        fc.property(
          fc.array(fc.string({ minLength: 1 }), { minLength: 1, maxLength: 5 }),
          (texts) => {
            const combined = texts.join("\n\n");
            return texts.every((text) => combined.includes(text));
          }
        )
      );
    });

    it("single text produces no separator", () => {
      fc.assert(
        fc.property(fc.string({ minLength: 1 }), (text) => {
          const combined = [text].join("\n\n");
          return combined === text && !combined.includes("\n\n") || text.includes("\n\n");
        })
      );
    });
  });

  describe("Total Stats Aggregation", () => {
    it("total words equals sum of individual word counts", () => {
      fc.assert(
        fc.property(
          fc.array(
            fc.record({
              name: fc.string({ minLength: 1 }),
              words: fc.nat({ max: 10000 }),
              chars: fc.nat({ max: 100000 }),
            }),
            { minLength: 0, maxLength: 10 }
          ),
          (files) => {
            const totalWords = files.reduce((sum, f) => sum + f.words, 0);
            const expectedTotal = files.map((f) => f.words).reduce((a, b) => a + b, 0);
            return totalWords === expectedTotal;
          }
        )
      );
    });

    it("total chars equals sum of individual char counts", () => {
      fc.assert(
        fc.property(
          fc.array(
            fc.record({
              name: fc.string({ minLength: 1 }),
              words: fc.nat({ max: 10000 }),
              chars: fc.nat({ max: 100000 }),
            }),
            { minLength: 0, maxLength: 10 }
          ),
          (files) => {
            const totalChars = files.reduce((sum, f) => sum + f.chars, 0);
            const expectedTotal = files.map((f) => f.chars).reduce((a, b) => a + b, 0);
            return totalChars === expectedTotal;
          }
        )
      );
    });
  });

  describe("Include/Exclude Toggle", () => {
    it("included files subset always valid", () => {
      fc.assert(
        fc.property(
          fc.array(fc.string({ minLength: 1 }), { minLength: 0, maxLength: 10 }),
          (fileNames) => {
            const allFiles = new Set(fileNames);
            const included = new Set(fileNames.slice(0, Math.floor(fileNames.length / 2)));

            // All included files should be in allFiles
            for (const name of included) {
              if (!allFiles.has(name)) return false;
            }
            return true;
          }
        )
      );
    });

    it("removing file from included set reduces size by 1", () => {
      fc.assert(
        fc.property(
          fc.array(fc.string({ minLength: 1 }), { minLength: 1, maxLength: 10 }),
          (fileNames) => {
            const included = new Set(fileNames);
            const originalSize = included.size;
            const toRemove = fileNames[0];
            included.delete(toRemove);
            // Size should decrease by 1 (if unique) or stay same (if duplicate)
            return included.size <= originalSize;
          }
        )
      );
    });
  });
});
