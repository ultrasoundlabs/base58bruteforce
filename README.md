# base58bruteforce

This is a simple GPU-accelerated CLI program for recovering proper case of Tron (Base58Check) addresses written in invalid (e.g. all lowercase) case.

> [!IMPORTANT]  
> Even though most of the code should work for all Base58Check inputs, some of it contains Tron-specific optimizations (e.g. enforcing first byte at 0x41, custom 21-byte-input SHA256 function). We haven't yet gotten our hands to making it universal. If you have, please push a PR.

## Building & Using

Prerequisites:
- Windows 10+ or Linux (only Ubuntu 22.04+ tested on our end; try at your own discretion)
- NVIDIA GPU with a supported CC version (see CMakeLists.txt; you can probably add your own, but no idea if it will work)
- MSVC or Clang (GCC not tested)
- CUDA Toolkit

To build:

```bash
chmod +x ./build.sh # only on Linux/Ubuntu
./build
```

To use:

```bash
./base58bruteforce <ambiguous-base58-string>
```

Example:

```bash
> ./base58bruteforce txykmwwahrpffepurnsqyngrji6vxxhkx9
TXYkMwWaHrPFfepurnsqYngrji6vxxHkX9
> ./base58bruteforce thesvzdaaxxxtdekaskptzndhugqnwuhwf
THesVZdaaXXxtdEkaskpTzndHugQnwUHwF
THEsvzdaAxXXTdEKasKptZNDhUgQNwUHwF
>
```

## Math & Benchmarks

The complexity of brute-forcing a Base58Check string with incorrect casing grows exponentially with the number of **ambiguous** characters it contains. Ambiguous characters are those that appear in both lower- and upper-case forms in the Base58 alphabet. The alphabet has nine unambiguous digits (1-9), 23 letters that exist in both cases (46 symbols total), and three unambiguous letters `o`, `i`, and `L` — whose case-mates are deliberately excluded. Together, 9 + 46 + 3 = 58 symbols make up Base58.

In an incorrect-casing scenario, every ambiguous letter doubles the number of candidate strings. A Base58 string with one ambiguous character (e.g., "a") therefore yields two possible variants. To discover which, if any, is valid, you must verify the checksum: compute SHA-256 twice over the entire byte sequence except the last four bytes, then compare the first four bytes of the hash to those final four bytes. Evaluating a single variant thus requires two SHA-256 hashes. With two ambiguous letters there are four variants; with three, sixteen; and so on.

Tron addresses are always 34-character Base58Check strings because the raw form is a 25-byte payload (0x41 "T" version byte + 20-byte EVM-style public-key hash + 4-byte double-SHA-256 checksum). Encoding 25 bytes in Base58 produces 34 characters:

$$
\boxed{
n=\left\lceil\frac{25 \cdot 8 - 1}{\log_{2}58}\right\rceil
= \left\lceil\frac{199}{\log_{2}58}\right\rceil
= 34
}
\quad\Longrightarrow\quad
58^{34} > 2^{199}
$$

Because the first byte is always 0x41 ("T"), a Tron address can contain at most 33 ambiguous characters. For the worst case, each of those 33 characters must be a letter other than `o`, `i`, or `L`; none of those three have case variants in Base58. Brute-forcing such an address would entail $2^{33} \times 2 \approx 16 \text{billion}$ SHA-256 computations. In practice, typical Tron addresses contain only 22-26 ambiguous letters, so the brute-force cost falls to roughly 8-132 million SHA-256 hashes.

Our implementation follows close complexity factor — a string with X ambiguous characters will take around 2x time (actually probably slightly more — our engineers are not good at GPU kernels) to brute-force than a string with X-1 ambiguous characters. **On RTX 4060,** an input string `thesvzdaaxxxtdekaskptzndhugqnwuhwf` (a Tron address with 33 ambiguous letters) takes ~19s to brute-force. Normal Tron addresses that have 22-26 ambiguous characters are nearly instant.

## Caveats

The more possible variations an invalid-case Tron address has, the higher the chance you will find **more than one** valid address. Here are the exact numbers:

$$
\boxed{%
\Pr[\text{collision}\mid m] =
1 - \bigl(1-2^{-40}\bigr)^{2^{m}-1}
}
$$

Where:

- $m$ — number of letters in the lowercase address (33 for "letter-only" addresses, 22-26 for a typical one)
- $2^{m}$ — total number of distinct upper/lower-case variants
- $2^{-40}$ — probability that **one specific** random 21-byte payload both matches the 4-byte checksum ($2^{-32}$) **and** starts with 0x41 (another $2^{-8}$)

In the worst case (33 ambiguous characters in an invalid-case address) this gives us an...

$$
1-\bigl(1-2^{-40}\bigr)^{2^{33}-1}\approx0.0078
$$

...**~0.78%** chance to get more than one valid Tron address. In normal Tron addresses, this is several orders of magnitude lower, but you don't really want to rely on this when sending a million dollars to the brute-forced address. So, if you integrate this somewhere, your internal logic should handle such situations.

## Why

This tool and research were made as part of the [Untron project](https://untron.finance). We'll be sharing more details about where we're using it soon.