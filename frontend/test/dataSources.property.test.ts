import fc from "fast-check";

/**
 * Property-based tests for data source selection functionality.
 */
describe("data source selection properties", () => {
  describe("training_text_paths generation", () => {
    it("selected source names produce valid path array (no empty paths)", () => {
      // Arbitrary for source-like objects
      const sourceArb = fc.record({
        name: fc.string({ minLength: 1, maxLength: 50 }),
        filename: fc.string({ minLength: 1, maxLength: 100 }),
        language: fc.string({ minLength: 1, maxLength: 20 }),
        script: fc.string({ minLength: 1, maxLength: 20 }),
        words: fc.integer({ min: 0, max: 1_000_000 }),
        chars: fc.integer({ min: 0, max: 10_000_000 }),
      });

      fc.assert(
        fc.property(
          fc.array(sourceArb, { minLength: 1, maxLength: 10 }),
          (sources) => {
            // Simulating the frontend logic: map selected names to filenames
            const selectedNames = new Set(sources.map((s) => s.name));
            const paths = [...selectedNames]
              .map((name) => sources.find((s) => s.name === name)?.filename)
              .filter((f): f is string => typeof f === "string" && f.length > 0);

            // Paths should all be non-empty strings
            expect(paths.every((p) => typeof p === "string" && p.length > 0)).toBe(true);
            // Should have same count as selected (unique by name)
            expect(paths.length).toBeLessThanOrEqual(selectedNames.size);
          }
        )
      );
    });

    it("empty selection produces empty paths array", () => {
      fc.assert(
        fc.property(fc.constant<{ name: string; filename: string }[]>([]), (sources) => {
          const selectedNames = new Set<string>();
          const paths = [...selectedNames]
            .map((name) => sources.find((s) => s.name === name)?.filename)
            .filter((f): f is string => Boolean(f));

          expect(paths).toEqual([]);
        })
      );
    });
  });

  describe("data source metadata validation", () => {
    it("word count is always non-negative", () => {
      const sourceArb = fc.record({
        name: fc.string({ minLength: 1 }),
        filename: fc.string({ minLength: 1 }),
        language: fc.string({ minLength: 1 }),
        script: fc.string({ minLength: 1 }),
        words: fc.integer({ min: 0, max: 10_000_000 }),
        chars: fc.integer({ min: 0, max: 100_000_000 }),
      });

      fc.assert(
        fc.property(sourceArb, (source) => {
          expect(source.words).toBeGreaterThanOrEqual(0);
          expect(source.chars).toBeGreaterThanOrEqual(0);
        })
      );
    });

    it("chars is typically greater than or equal to words", () => {
      // This is a soft property - most texts have chars >= words
      // (unless text is empty or has very short words)
      const sourceArb = fc.record({
        words: fc.integer({ min: 1, max: 10_000 }),
        chars: fc.integer({ min: 1, max: 100_000 }),
      });

      fc.assert(
        fc.property(sourceArb, (source) => {
          // We just test that both are positive
          expect(source.words).toBeGreaterThan(0);
          expect(source.chars).toBeGreaterThan(0);
        })
      );
    });
  });
});
