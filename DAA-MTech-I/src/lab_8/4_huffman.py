"""
Assignment 4:
Write a program to implement file compression (and decompression) using Huffman’s algorithm.
"""

import heapq
import os
import struct
import sys
from collections import Counter
from pathlib import Path
import tempfile
from typing import Dict, Optional, Tuple

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.lab_8.bit_io import BitReader, BitWriter


MAGIC_HEADER = b"HUFF"


class HuffmanNode:
    """Represents a node in the Huffman Tree."""

    def __init__(
        self,
        symbol: Optional[int],
        freq: int,
        left: Optional["HuffmanNode"] = None,
        right: Optional["HuffmanNode"] = None,
    ):
        self.symbol = symbol  # byte integer (0-255) or None for internal nodes
        self.freq = freq
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __lt__(self, other: "HuffmanNode") -> bool:
        # Tie-break on frequency first; if equal, maintain stable deterministic ordering
        if self.freq != other.freq:
            return self.freq < other.freq
        s1 = self.symbol if self.symbol is not None else -1
        s2 = other.symbol if other.symbol is not None else -1
        return s1 < s2


def build_huffman_tree(frequencies: Dict[int, int]) -> Optional[HuffmanNode]:
    """
    Builds a Huffman Tree from symbol frequencies using a min-heap.
    Time Complexity: O(k log k), where k is the number of unique symbols (k <= 256 for bytes).
    """
    if not frequencies:
        return None

    heap = [HuffmanNode(symbol=sym, freq=freq) for sym, freq in frequencies.items()]
    heapq.heapify(heap)

    # Edge case: single unique character
    if len(heap) == 1:
        root = heapq.heappop(heap)
        return HuffmanNode(symbol=None, freq=root.freq, left=root)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(
            symbol=None,
            freq=node1.freq + node2.freq,
            left=node1,
            right=node2,
        )
        heapq.heappush(heap, merged)

    return heap[0]


def generate_codes(root: Optional[HuffmanNode]) -> Dict[int, str]:
    """
    Generates prefix-free variable-length binary codes by traversing the Huffman tree.
    """
    codes: Dict[int, str] = {}

    def _traverse(node: Optional[HuffmanNode], current_code: str):
        if node is None:
            return
        if node.is_leaf():
            # If root has only one child, use '0'
            codes[node.symbol] = current_code if current_code else "0"
            return
        _traverse(node.left, current_code + "0")
        _traverse(node.right, current_code + "1")

    _traverse(root, "")
    return codes


def compress_bytes(raw_data: bytes) -> bytes:
    """
    Compresses raw bytes using Huffman Coding.
    Serialized format:
    - 4 bytes: Magic bytes b"HUFF"
    - 2 bytes: Number of unique symbols k (unsigned short)
    - For each symbol: 1 byte (symbol) + 4 bytes (frequency as unsigned int)
    - 1 byte: Padding bits count (0 to 7)
    - Variable: Encoded bitstream
    """
    if not raw_data:
        # Empty input representation
        return MAGIC_HEADER + struct.pack(">H", 0)

    # 1. Calculate symbol frequencies
    freq_counter = Counter(raw_data)
    unique_symbols = len(freq_counter)

    # 2. Build Huffman tree and prefix codes
    root = build_huffman_tree(freq_counter)
    codes = generate_codes(root)

    # 3. Encode data bit-by-bit
    writer = BitWriter()
    for byte in raw_data:
        writer.write_bits(codes[byte])

    encoded_payload, padding_bits = writer.flush()

    # 4. Serialize header
    header_parts = [MAGIC_HEADER, struct.pack(">H", unique_symbols)]
    for sym, freq in freq_counter.items():
        header_parts.append(struct.pack(">BI", sym, freq))
    header_parts.append(struct.pack(">B", padding_bits))

    return b"".join(header_parts) + encoded_payload


def decompress_bytes(compressed_data: bytes) -> bytes:
    """
    Decompresses data compressed by compress_bytes.
    """
    if len(compressed_data) < 6:
        raise ValueError("Invalid compressed data: too short for header.")

    # 1. Verify magic header
    if compressed_data[:4] != MAGIC_HEADER:
        raise ValueError("Invalid format: magic header b'HUFF' missing.")

    num_symbols = struct.unpack_from(">H", compressed_data, 4)[0]
    if num_symbols == 0:
        return b""

    # 2. Parse frequency table
    offset = 6
    frequencies: Dict[int, int] = {}
    for _ in range(num_symbols):
        sym, freq = struct.unpack_from(">BI", compressed_data, offset)
        frequencies[sym] = freq
        offset += 5

    padding_bits = struct.unpack_from(">B", compressed_data, offset)[0]
    offset += 1

    payload = compressed_data[offset:]

    # 3. Rebuild Huffman tree
    root = build_huffman_tree(frequencies)
    if root is None:
        return b""

    # Calculate expected uncompressed length
    total_bytes_expected = sum(frequencies.values())

    # 4. Decode payload bitstream
    reader = BitReader(payload, padding=padding_bits)
    decoded = bytearray()
    curr_node = root

    while reader.has_next_bit() and len(decoded) < total_bytes_expected:
        bit = reader.read_bit()
        curr_node = curr_node.left if bit == 0 else curr_node.right

        if curr_node.is_leaf():
            decoded.append(curr_node.symbol)
            curr_node = root

    return bytes(decoded)


# ==============================================================================
# FILE COMPRESSION & DECOMPRESSION INTERFACE
# ==============================================================================
def compress_file(input_path: str | Path, output_path: str | Path) -> dict:
    """Compresses an input file to a Huffman-encoded output file."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, "rb") as f:
        raw_bytes = f.read()

    compressed_bytes = compress_bytes(raw_bytes)

    with open(output_path, "wb") as f:
        f.write(compressed_bytes)

    orig_size = len(raw_bytes)
    comp_size = len(compressed_bytes)
    ratio = (1.0 - comp_size / orig_size) * 100 if orig_size > 0 else 0.0

    return {
        "original_size": orig_size,
        "compressed_size": comp_size,
        "savings_percentage": ratio,
    }


def decompress_file(compressed_path: str | Path, output_path: str | Path) -> dict:
    """Decompresses a Huffman-encoded file to restore the original file."""
    compressed_path = Path(compressed_path)
    output_path = Path(output_path)

    with open(compressed_path, "rb") as f:
        compressed_bytes = f.read()

    restored_bytes = decompress_bytes(compressed_bytes)

    with open(output_path, "wb") as f:
        f.write(restored_bytes)

    return {
        "restored_size": len(restored_bytes),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Assignment 4: Huffman File Compression and Decompression")
    print("=" * 80)

    sample_text = (
        "In computer science and information theory, a Huffman code is a particular type "
        "of optimal prefix code that is commonly used for lossless data compression. "
        "The process of finding or using such a code proceeds by means of Huffman coding, "
        "an algorithm developed by David A. Huffman while he was a Sc.D. student at MIT, "
        "and published in the 1952 paper 'A Method for the Construction of Minimum-Redundancy Codes'.\n"
        "Repeated patterns: AAAAAABBBBBCCCCCDDDDEEEFFFFFGGGGHHHHIIIIJJJJKKKKLLLLMMMM\n"
        "Numbers and symbols: 1234567890 !@#$%^&*()_+~`\n"
    ) * 15

    # Run file roundtrip test inside an auto-cleaning temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        orig_file = temp_path / "sample_text.txt"
        comp_file = temp_path / "sample_text.huff"
        restored_file = temp_path / "sample_text_restored.txt"

        # Write sample file
        with open(orig_file, "w", encoding="utf-8") as f:
            f.write(sample_text)

        print(f"Sample File Created    : {orig_file.name}")
        comp_stats = compress_file(orig_file, comp_file)
        print(f"Compressed File Created: {comp_file.name}")
        print(f" - Original Size        : {comp_stats['original_size']} bytes")
        print(f" - Compressed Size      : {comp_stats['compressed_size']} bytes")
        print(f" - Space Savings        : {comp_stats['savings_percentage']:.2f}%")

        # Decompress file
        decomp_stats = decompress_file(comp_file, restored_file)
        print(f"Restored File Created  : {restored_file.name}")
        print(f" - Restored Size        : {decomp_stats['restored_size']} bytes")

        # Verify lossless roundtrip bit-for-bit
        with open(orig_file, "rb") as f1, open(restored_file, "rb") as f2:
            orig_data = f1.read()
            restored_data = f2.read()
            assert orig_data == restored_data, "Decompression verification failed: content mismatch!"

        print("Verification Passed: Decompressed content is bit-for-bit identical to original!")

    # Binary test (simulating arbitrary binary file)
    binary_data = bytes([i % 256 for i in range(1000)] + [42] * 2000)
    bin_comp = compress_bytes(binary_data)
    bin_decomp = decompress_bytes(bin_comp)
    assert binary_data == bin_decomp, "Binary decompression mismatch!"
    print(f"Binary Test Passed     : {len(binary_data)} bytes -> {len(bin_comp)} bytes -> verified exact match!")
    print("=" * 80)
