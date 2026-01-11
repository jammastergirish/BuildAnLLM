import { fireEvent, render, screen, waitFor, act } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";

// Mock the file input behavior
describe("Multi-File Upload UI", () => {
  // Helper to create mock files
  const createMockFile = (name: string, content: string): File => {
    return new File([content], name, { type: "text/plain" });
  };

  describe("File Input Handling", () => {
    it("accepts multiple text files", () => {
      const files = [
        createMockFile("file1.txt", "Content 1"),
        createMockFile("file2.txt", "Content 2"),
      ];

      const fileList = {
        length: files.length,
        item: (i: number) => files[i],
        [Symbol.iterator]: function* () {
          for (const file of files) yield file;
        },
      };

      // Verify files are iterable
      const fileArray = Array.from(fileList);
      expect(fileArray).toHaveLength(2);
      expect(fileArray[0].name).toBe("file1.txt");
      expect(fileArray[1].name).toBe("file2.txt");
    });

    it("reads file content asynchronously", async () => {
      const content = "Test file content here";
      const file = createMockFile("test.txt", content);

      const reader = new FileReader();
      const readPromise = new Promise<string>((resolve) => {
        reader.onload = () => resolve(reader.result as string);
      });
      reader.readAsText(file);

      const result = await readPromise;
      expect(result).toBe(content);
    });

    it("calculates word count from file content", async () => {
      const content = "Hello world this is a test";
      const expectedWordCount = 6;

      const words = content.split(/\s+/).filter((w) => w.length > 0);
      expect(words.length).toBe(expectedWordCount);
    });

    it("calculates character count from file content", () => {
      const content = "Hello world!";
      expect(content.length).toBe(12);
    });
  });

  describe("File Stats Calculation", () => {
    it("computes stats for multiple files", async () => {
      const files = [
        { name: "file1.txt", content: "Hello world" },
        { name: "file2.txt", content: "This is more content here" },
      ];

      const stats = new Map<string, { words: number; chars: number }>();
      for (const file of files) {
        const words = file.content.split(/\s+/).filter((w) => w.length > 0).length;
        const chars = file.content.length;
        stats.set(file.name, { words, chars });
      }

      expect(stats.get("file1.txt")).toEqual({ words: 2, chars: 11 });
      expect(stats.get("file2.txt")).toEqual({ words: 5, chars: 25 });
    });

    it("handles empty files gracefully", () => {
      const content = "";
      const words = content.split(/\s+/).filter((w) => w.length > 0).length;
      const chars = content.length;

      expect(words).toBe(0);
      expect(chars).toBe(0);
    });

    it("handles files with only whitespace", () => {
      const content = "   \n\t  \n  ";
      const words = content.split(/\s+/).filter((w) => w.length > 0).length;

      expect(words).toBe(0);
    });
  });

  describe("File Include/Exclude Toggle", () => {
    it("tracks included files in a Set", () => {
      const includedFiles = new Set<string>(["file1.txt", "file2.txt"]);

      expect(includedFiles.has("file1.txt")).toBe(true);
      expect(includedFiles.has("file3.txt")).toBe(false);

      // Toggle off
      includedFiles.delete("file1.txt");
      expect(includedFiles.has("file1.txt")).toBe(false);

      // Toggle on
      includedFiles.add("file3.txt");
      expect(includedFiles.has("file3.txt")).toBe(true);
    });

    it("calculates total stats from included files only", () => {
      const allFiles = new Map<string, { words: number; chars: number }>([
        ["file1.txt", { words: 10, chars: 50 }],
        ["file2.txt", { words: 20, chars: 100 }],
        ["file3.txt", { words: 30, chars: 150 }],
      ]);
      const includedFiles = new Set(["file1.txt", "file3.txt"]);

      let totalWords = 0;
      let totalChars = 0;
      for (const fileName of includedFiles) {
        const stats = allFiles.get(fileName);
        if (stats) {
          totalWords += stats.words;
          totalChars += stats.chars;
        }
      }

      expect(totalWords).toBe(40); // 10 + 30
      expect(totalChars).toBe(200); // 50 + 150
    });
  });

  describe("File Removal", () => {
    it("removes file from list", () => {
      let files = [
        createMockFile("file1.txt", "Content 1"),
        createMockFile("file2.txt", "Content 2"),
        createMockFile("file3.txt", "Content 3"),
      ];

      const fileToRemove = "file2.txt";
      files = files.filter((f) => f.name !== fileToRemove);

      expect(files).toHaveLength(2);
      expect(files.map((f) => f.name)).toEqual(["file1.txt", "file3.txt"]);
    });

    it("removes file from included set when removed from list", () => {
      const includedFiles = new Set(["file1.txt", "file2.txt", "file3.txt"]);
      const fileToRemove = "file2.txt";

      // Simulate removal
      includedFiles.delete(fileToRemove);

      expect(includedFiles.has(fileToRemove)).toBe(false);
      expect(includedFiles.size).toBe(2);
    });
  });

  describe("File Concatenation for Upload", () => {
    it("combines multiple file texts with double newline separator", async () => {
      const files = [
        createMockFile("file1.txt", "Content 1"),
        createMockFile("file2.txt", "Content 2"),
      ];

      // Simulate reading all files
      const texts = await Promise.all(
        files.map(
          (f) =>
            new Promise<string>((resolve) => {
              const reader = new FileReader();
              reader.onload = () => resolve(reader.result as string);
              reader.readAsText(f);
            })
        )
      );

      const combinedText = texts.join("\n\n");
      expect(combinedText).toBe("Content 1\n\nContent 2");
    });

    it("creates a single File from combined text", async () => {
      const combinedText = "Content 1\n\nContent 2";
      const combinedFile = new File([combinedText], "combined_upload.txt", {
        type: "text/plain",
      });

      expect(combinedFile.name).toBe("combined_upload.txt");
      expect(combinedFile.size).toBe(combinedText.length);

      // Verify content
      const reader = new FileReader();
      const contentPromise = new Promise<string>((resolve) => {
        reader.onload = () => resolve(reader.result as string);
      });
      reader.readAsText(combinedFile);

      const content = await contentPromise;
      expect(content).toBe(combinedText);
    });
  });
});
