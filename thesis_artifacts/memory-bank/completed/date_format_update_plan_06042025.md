# Plan: Update Memory Bank Date Formats

**Date:** 06-04-2025

**Goal:**
Update all timestamps within the content of the five core Memory Bank files to the consistent `DD-MM-YYYY HH:MM:SS` format, correcting for entries where the month and day were originally flipped (specifically those appearing as May or June 2025).

**Target Files:**
1.  `memory-bank/productContext.md`
2.  `memory-bank/activeContext.md`
3.  `memory-bank/systemPatterns.md`
4.  `memory-bank/decisionLog.md`
5.  `memory-bank/progress.md`

**Required Format Change:**
*   Correct `YYYY-DD-MM HH:MM:SS` (where MM was 05 or 06) to `DD-MM-YYYY HH:MM:SS`.
*   Convert correct `YYYY-MM-DD HH:MM:SS` (where MM was 04) to `DD-MM-YYYY HH:MM:SS`.

**Tool:**
The `search_and_replace` tool will be used, performing **two separate operations** for each file.

**Detailed Steps (Two Operations per File):**

For *each* of the 5 target files, perform these two operations sequentially:

1.  **Operation 1: Correct Flipped May/June Dates**
    *   **Purpose:** Find timestamps from 2025 appearing as May or June (which are actually April 5th/6th with flipped day/month) and format them correctly as `DD-MM-YYYY HH:MM:SS`.
    *   **Search Regex:** `\[(2025)-(0[56])-(\d{2}) (\d{2}:\d{2}:\d{2})\]`
    *   **Replace Pattern:** `[$3-$2-$1 $4]`
    *   **Example:** `[2025-06-04 17:43:00]` becomes `[04-06-2025 17:43:00]`

2.  **Operation 2: Convert Correct April Dates**
    *   **Purpose:** Find timestamps from April 2025 (assumed to be correctly formatted as `YYYY-MM-DD`) and convert them to `DD-MM-YYYY HH:MM:SS`.
    *   **Search Regex:** `\[(2025)-(04)-(\d{2}) (\d{2}:\d{2}:\d{2})\]`
    *   **Replace Pattern:** `[$3-$2-$1 $4]`
    *   **Example:** `[2025-04-01 01:06:28]` becomes `[01-04-2025 01:06:28]`

**Execution Flow:**
```mermaid
graph TD
    A[Start] --> B{File 1: productContext.md};
    B --> C1[Apply Op 1: Fix Flipped May/June];
    C1 --> C2[Apply Op 2: Convert April];
    C2 --> D{File 2: activeContext.md};
    D --> E1[Apply Op 1: Fix Flipped May/June];
    E1 --> E2[Apply Op 2: Convert April];
    E2 --> F{File 3: systemPatterns.md};
    F --> G1[Apply Op 1: Fix Flipped May/June];
    G1 --> G2[Apply Op 2: Convert April];
    G2 --> H{File 4: decisionLog.md};
    H --> I1[Apply Op 1: Fix Flipped May/June];
    I1 --> I2[Apply Op 2: Convert April];
    I2 --> J{File 5: progress.md};
    J --> K1[Apply Op 1: Fix Flipped May/June];
    K1 --> K2[Apply Op 2: Convert April];
    K2 --> L[End];