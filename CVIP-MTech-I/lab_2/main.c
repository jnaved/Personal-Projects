/**
 * ====================================================================================================
 * COURSE: CVIP (Computer Vision and Image Processing) - M.Tech
 * LAB 2 : NETPBM IMAGE FORMAT IDENTIFICATION, HEADER PARSING, DATA SIZE CALCULATION & VERIFICATION
 * ====================================================================================================
 *
 * Objectives Implemented:
 *  1. Format Identification  : Read binary header, extract magic number (P5, P6, P2, P3), explain representation.
 *  2. Header Parsing         : Extract Width, Height, MaxVal with robust whitespace and '#' comment handling.
 *  3. Pixel Count            : Compute total pixels = Width * Height.
 *  4. Bytes Per Pixel        : Determine sample depth (1 vs 2 bytes) and pixel depth (PGM=1/2 bytes, PPM=3/6 bytes).
 *  5. Expected Data Size     : Calculate and display expected raster payload size.
 *  6. Image Data Read        : Dynamically allocate memory and read exact payload bytes from file.
 *  7. File Integrity Check   : Compare expected payload size with actual file size minus header.
 *  8. First 10 Pixels        : Display first 10 pixels (scalar for PGM, RGB triplets for PPM, 16-bit explanation).
 *  9. Extension vs Format    : Test PGM vs PPM and prove header magic number is authoritative over extension.
 * 10. Critical-Thinking      : Perform 640x480 analytical calculations for MaxVal=255 vs MaxVal=65535.
 * ====================================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/* Structure to store complete metadata, size metrics, and verification status */
typedef struct {
    long long total_pixels;          /* Width * Height */
    int channels;                    /* 1 for Grayscale (PGM), 3 for RGB (PPM) */
    int bytes_per_sample;            /* 1 if MaxVal <= 255, 2 if MaxVal > 255 */
    int bytes_per_pixel;             /* channels * bytes_per_sample */
    long long expected_payload_size; /* total_pixels * bytes_per_pixel */
    long long header_size_bytes;     /* Byte offset where raster begins */
    long long total_file_size_bytes; /* Total file size on disk */
    long long actual_payload_size;   /* total_file_size_bytes - header_size_bytes */
    size_t bytes_successfully_read;  /* Bytes read by fread */
    int is_verified;                 /* 1 if actual == expected, 0 if corrupted/truncated */
} ImageMetrics;

/* Unified structure representing a Netpbm Image (PGM or PPM) */
typedef struct {
    char filename[256];
    char magic[4];                   /* "P5", "P6", "P2", "P3" */
    int format_id;                   /* 5 for P5, 6 for P6, 2 for P2, 3 for P3 */
    int width;
    int height;
    int max_val;
    int channels;                    /* 1 for Grayscale, 3 for RGB */
    int bytes_per_sample;            /* 1 or 2 */
    int bytes_per_pixel;             /* channels * bytes_per_sample */
    unsigned char *raw_data;         /* Dynamically allocated raw byte payload */
    ImageMetrics metrics;            /* Embedded self-contained metrics struct (SOLID/DRY) */
} NetpbmImage;

/* Function Prototypes */
static void skip_comments_and_whitespace(FILE *fp);
static long long get_file_size(const char *filepath);
static const char* resolve_input_path(const char *rel_path, char *buf, size_t buf_size);
NetpbmImage* read_netpbm_image(const char *filepath);
void free_netpbm_image(NetpbmImage *img);
void display_format_explanation(const char *magic);
void display_complete_image_report(const NetpbmImage *img);
void display_first_ten_pixels(const NetpbmImage *img);
void perform_extension_vs_format_investigation(const char *pgm_path, const char *ppm_path);
void perform_critical_thinking_analysis(void);


/**
 * Helper to skip all whitespace characters and '#' comment lines in Netpbm headers.
 */
static void skip_comments_and_whitespace(FILE *fp) {
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '#') {
            /* Skip entire comment line until newline */
            while ((ch = fgetc(fp)) != EOF && ch != '\n' && ch != '\r');
        } else if (!isspace(ch)) {
            /* Put back first non-whitespace character */
            ungetc(ch, fp);
            break;
        }
    }
}

/**
 * Helper to obtain the total size of a file in bytes from disk.
 */
static long long get_file_size(const char *filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long long sz = ftell(fp);
    fclose(fp);
    return sz;
}

/**
 * Robust path resolver for root, lab_2/, and lab_1/ directories
 */
static const char* resolve_input_path(const char *rel_path, char *buf, size_t buf_size) {
    FILE *f = fopen(rel_path, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", rel_path); return buf; }

    char candidate[512];
    snprintf(candidate, sizeof(candidate), "images/%s", rel_path);
    f = fopen(candidate, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", candidate); return buf; }

    snprintf(candidate, sizeof(candidate), "../images/%s", rel_path);
    f = fopen(candidate, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", candidate); return buf; }

    snprintf(candidate, sizeof(candidate), "lab_2/images/%s", rel_path);
    f = fopen(candidate, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", candidate); return buf; }

    snprintf(candidate, sizeof(candidate), "../lab_2/images/%s", rel_path);
    f = fopen(candidate, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", candidate); return buf; }

    snprintf(candidate, sizeof(candidate), "lab_1/images/%s", rel_path);
    f = fopen(candidate, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", candidate); return buf; }

    snprintf(candidate, sizeof(candidate), "../lab_1/images/%s", rel_path);
    f = fopen(candidate, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", candidate); return buf; }

    snprintf(buf, buf_size, "%s", rel_path);
    return buf;
}

/**
 * Open a PGM/PPM file in binary mode, parse header, calculate expected size,
 * read raster data, and verify file integrity.
 */
NetpbmImage* read_netpbm_image(const char *filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Unable to open image file: '%s'\n", filepath);
        return NULL;
    }

    /* 1. Identify Magic Number */
    char magic[4] = {0};
    if (fscanf(fp, "%2s", magic) != 1) {
        fprintf(stderr, "[ERROR] Failed to read magic number from '%s'\n", filepath);
        fclose(fp);
        return NULL;
    }

    int format_id = 0;
    int channels = 1;
    if (strcmp(magic, "P5") == 0) {
        format_id = 5;
        channels = 1;  /* Grayscale Binary */
    } else if (strcmp(magic, "P6") == 0) {
        format_id = 6;
        channels = 3;  /* RGB Truecolor Binary */
    } else if (strcmp(magic, "P2") == 0) {
        format_id = 2;
        channels = 1;  /* Grayscale ASCII */
    } else if (strcmp(magic, "P3") == 0) {
        format_id = 3;
        channels = 3;  /* RGB Truecolor ASCII */
    } else {
        fprintf(stderr, "[ERROR] Unsupported magic number '%s' in '%s'. Supported: P5, P6, P2, P3.\n", magic, filepath);
        fclose(fp);
        return NULL;
    }

    /* 2. Read Header Fields (Width, Height, MaxVal) with comment handling */
    skip_comments_and_whitespace(fp);
    int width = 0, height = 0, max_val = 0;
    if (fscanf(fp, "%d", &width) != 1 || width <= 0) {
        fprintf(stderr, "[ERROR] Invalid width in '%s'\n", filepath);
        fclose(fp);
        return NULL;
    }

    skip_comments_and_whitespace(fp);
    if (fscanf(fp, "%d", &height) != 1 || height <= 0) {
        fprintf(stderr, "[ERROR] Invalid height in '%s'\n", filepath);
        fclose(fp);
        return NULL;
    }

    skip_comments_and_whitespace(fp);
    if (fscanf(fp, "%d", &max_val) != 1 || max_val <= 0 || max_val > 65535) {
        fprintf(stderr, "[ERROR] Invalid MaxVal (%d) in '%s'. Must be in [1, 65535].\n", max_val, filepath);
        fclose(fp);
        return NULL;
    }

    /* In Netpbm binary formats, exactly one whitespace character separates MaxVal from raw data */
    int sep = fgetc(fp);
    if (sep == EOF) {
        fprintf(stderr, "[ERROR] Unexpected EOF immediately after header in '%s'\n", filepath);
        fclose(fp);
        return NULL;
    }
    long long header_end_pos = ftell(fp);

    /* 3. Allocate NetpbmImage Structure */
    NetpbmImage *img = (NetpbmImage*)malloc(sizeof(NetpbmImage));
    if (!img) {
        fprintf(stderr, "[ERROR] Memory allocation failed for NetpbmImage struct\n");
        fclose(fp);
        return NULL;
    }
    memset(img, 0, sizeof(NetpbmImage));

    strncpy(img->filename, filepath, sizeof(img->filename) - 1);
    img->filename[sizeof(img->filename) - 1] = '\0';
    snprintf(img->magic, sizeof(img->magic), "%s", magic);
    img->format_id = format_id;
    img->width = width;
    img->height = height;
    img->max_val = max_val;
    img->channels = channels;

    /* 4. Determine Bytes Per Sample, Bytes Per Pixel, and Expected Data Size */
    int bytes_per_sample = (max_val <= 255) ? 1 : 2;
    int bytes_per_pixel = channels * bytes_per_sample;
    long long total_pixels = (long long)width * height;
    long long expected_payload = total_pixels * bytes_per_pixel;

    img->bytes_per_sample = bytes_per_sample;
    img->bytes_per_pixel = bytes_per_pixel;

    img->metrics.total_pixels = total_pixels;
    img->metrics.channels = channels;
    img->metrics.bytes_per_sample = bytes_per_sample;
    img->metrics.bytes_per_pixel = bytes_per_pixel;
    img->metrics.expected_payload_size = expected_payload;
    img->metrics.header_size_bytes = header_end_pos;

    /* 5. Check Disk File Size and Compute Actual Available Data */
    long long file_size = get_file_size(filepath);
    img->metrics.total_file_size_bytes = file_size;
    img->metrics.actual_payload_size = file_size - header_end_pos;

    /* 6. Read Raw Pixel Payload */
    if (format_id == 5 || format_id == 6) {
        /* Binary raw read (P5 or P6) */
        img->raw_data = (unsigned char*)malloc(expected_payload * sizeof(unsigned char));
        if (!img->raw_data) {
            fprintf(stderr, "[ERROR] Memory allocation failed for %lld pixel bytes\n", expected_payload);
            free(img);
            fclose(fp);
            return NULL;
        }

        size_t read_bytes = fread(img->raw_data, sizeof(unsigned char), expected_payload, fp);
        img->metrics.bytes_successfully_read = read_bytes;

        /* Verify data integrity */
        if ((long long)read_bytes == expected_payload && img->metrics.actual_payload_size == expected_payload) {
            img->metrics.is_verified = 1;
        } else {
            img->metrics.is_verified = 0;
        }
    } else {
        /* ASCII Format (P2 or P3) */
        long long total_samples = total_pixels * channels;
        img->raw_data = (unsigned char*)malloc(total_samples * sizeof(unsigned char));
        if (!img->raw_data) {
            fprintf(stderr, "[ERROR] Memory allocation failed for ASCII samples\n");
            free(img);
            fclose(fp);
            return NULL;
        }

        long long samples_read = 0;
        for (long long i = 0; i < total_samples; i++) {
            int val;
            if (fscanf(fp, "%d", &val) != 1) {
                fprintf(stderr, "[WARNING] Premature EOF reading ASCII samples at index %lld\n", i);
                break;
            }
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            img->raw_data[i] = (unsigned char)val;
            samples_read++;
        }
        img->metrics.bytes_successfully_read = (size_t)samples_read;
        img->metrics.is_verified = (samples_read == total_samples) ? 1 : 0;
    }

    fclose(fp);
    return img;
}

/**
 * Free memory occupied by NetpbmImage
 */
void free_netpbm_image(NetpbmImage *img) {
    if (img) {
        if (img->raw_data) {
            free(img->raw_data);
            img->raw_data = NULL;
        }
        free(img);
    }
}

/**
 * Explanation of the Magic Number in Netpbm format
 */
void display_format_explanation(const char *magic) {
    printf("  [Netpbm Magic Number Explanation]:\n");
    if (strcmp(magic, "P5") == 0) {
        printf("    * Magic Identifier: 'P5' represents Portable Graymap (PGM) in RAW BINARY format.\n");
        printf("    * Color Model     : Grayscale (1 component / channel per pixel: Intensity).\n");
        printf("    * Encoding Mode   : Uncompressed binary bytes for high storage efficiency.\n");
    } else if (strcmp(magic, "P6") == 0) {
        printf("    * Magic Identifier: 'P6' represents Portable Pixmap (PPM) in RAW BINARY format.\n");
        printf("    * Color Model     : 24-bit Truecolor RGB (3 components / channels per pixel: Red, Green, Blue).\n");
        printf("    * Encoding Mode   : Uncompressed binary bytes (RGBRGB... sequence).\n");
    } else if (strcmp(magic, "P2") == 0) {
        printf("    * Magic Identifier: 'P2' represents Portable Graymap (PGM) in PLAIN ASCII format.\n");
        printf("    * Color Model     : Grayscale (1 channel, decimal numbers formatted as ASCII text).\n");
    } else if (strcmp(magic, "P3") == 0) {
        printf("    * Magic Identifier: 'P3' represents Portable Pixmap (PPM) in PLAIN ASCII format.\n");
        printf("    * Color Model     : RGB (3 channels, space/newline-separated decimal ASCII text).\n");
    }
}

/**
 * Display complete analysis, calculated sizes, and verification report
 */
void display_complete_image_report(const NetpbmImage *img) {
    if (!img) return;
    const ImageMetrics *m = &img->metrics;

    printf("\n========================================================================================\n");
    printf("                        NETPBM IMAGE ANALYSIS & VERIFICATION REPORT                     \n");
    printf("========================================================================================\n");
    printf("  [1] File Identification & Header Ground Truth:\n");
    printf("      * File Path               : %s\n", img->filename);
    printf("      * Magic Number            : %s\n", img->magic);
    display_format_explanation(img->magic);

    printf("\n  [2] Geometric Dimensions & Header Fields:\n");
    printf("      * Image Width (W)         : %d pixels\n", img->width);
    printf("      * Image Height (H)        : %d pixels\n", img->height);
    printf("      * Max Intensity (MaxVal)  : %d\n", img->max_val);
    printf("      * Total Pixels (W x H)    : %lld pixels (%.2f Megapixels)\n",
           m->total_pixels, (double)m->total_pixels / 1e6);

    printf("\n  [3] Pixel Storage Architecture & Precision:\n");
    printf("      * Color Channels          : %d (%s)\n",
           m->channels, (m->channels == 1 ? "1 Grayscale Channel" : "3 Channels: Red, Green, Blue"));
    printf("      * Bytes Per Sample        : %d byte(s) %s\n",
           m->bytes_per_sample,
           (img->max_val <= 255 ? "[MaxVal <= 255 -> 8-bit precision]" : "[MaxVal > 255 -> 16-bit precision]"));
    printf("      * Total Bytes Per Pixel   : %d byte(s)/pixel (Channels: %d x BytesPerSample: %d)\n",
           m->bytes_per_pixel, m->channels, m->bytes_per_sample);

    printf("\n  [4] Mathematical Calculation of Expected Image-Data Size:\n");
    printf("      * Formula: Expected Size = Width * Height * Channels * BytesPerSample\n");
    printf("      * Step-by-Step Arithmetic:\n");
    printf("            = (%d pixels) * (%d pixels) * (%d channels) * (%d byte%s)\n",
           img->width, img->height, m->channels, m->bytes_per_sample, (m->bytes_per_sample > 1 ? "s" : ""));
    printf("            = %lld pixels * %d bytes/pixel\n", m->total_pixels, m->bytes_per_pixel);
    printf("            = %lld bytes (%.2f MB / %.2f KB)\n",
           m->expected_payload_size,
           (double)m->expected_payload_size / (1024.0 * 1024.0),
           (double)m->expected_payload_size / 1024.0);

    printf("\n  [5] File Verification & Structural Integrity Analysis:\n");
    printf("      * Total File Size on Disk : %lld bytes\n", m->total_file_size_bytes);
    printf("      * Header Offset / Length  : %lld bytes\n", m->header_size_bytes);
    printf("      * Actual Data In File     : %lld bytes (Total Size - Header Length)\n", m->actual_payload_size);
    printf("      * Bytes Read into Memory  : %zu bytes\n", m->bytes_successfully_read);

    if (m->is_verified) {
        printf("      * INTEGRITY STATUS        : [VERIFIED PERFECT - 100%% MATCH]\n");
        printf("        -> Expected raster payload (%lld bytes) EXACTLY matches actual available data.\n",
               m->expected_payload_size);
    } else {
        printf("      * INTEGRITY STATUS        : [FAILED / MISMATCH DETECTED]\n");
        printf("        -> Warning: Expected (%lld bytes) does NOT match actual available (%lld bytes)!\n",
               m->expected_payload_size, m->actual_payload_size);
        printf("        -> The file may be incomplete, truncated, corrupted, or incorrectly interpreted.\n");
    }
    printf("========================================================================================\n");
}

/**
 * Display first 10 pixels (scalar for PGM, RGB triplets for PPM, 16-bit explanation)
 */
void display_first_ten_pixels(const NetpbmImage *img) {
    if (!img || !img->raw_data) return;

    printf("\n========================================================================================\n");
    printf("                        INITIAL PIXEL VALUE ANALYSIS (FIRST 10 PIXELS)                  \n");
    printf("                        File: %s (Format: %s)\n", img->filename, img->magic);
    printf("========================================================================================\n");

    int num_pixels_to_show = (img->metrics.total_pixels < 10) ? (int)img->metrics.total_pixels : 10;

    if (img->channels == 1) {
        /* PGM Grayscale Image */
        printf(" Index | Pixel Coordinate (X, Y) | Intensity Sample Value | Normalized [0.0 - 1.0]\n");
        printf("-------+-------------------------+------------------------+-----------------------\n");
        for (int i = 0; i < num_pixels_to_show; i++) {
            int x = i % img->width;
            int y = i / img->width;
            if (img->bytes_per_sample == 1) {
                unsigned char val = img->raw_data[i];
                printf("  #%02d  |   (%4d, %4d)           |  %3d / %-3d            |       %.4f\n",
                       i, x, y, val, img->max_val, (double)val / (double)img->max_val);
            } else {
                /* 16-bit PGM Sample: Two bytes per sample in Big-Endian order */
                unsigned char msb = img->raw_data[2 * i];
                unsigned char lsb = img->raw_data[2 * i + 1];
                unsigned short val = (unsigned short)((msb << 8) | lsb);
                printf("  #%02d  |   (%4d, %4d)           |  %5d / %-5d        |       %.4f\n",
                       i, x, y, val, img->max_val, (double)val / (double)img->max_val);
            }
        }
    } else {
        /* PPM Truecolor RGB Image */
        printf(" Index | Pixel (X, Y) |   Red (R)   |  Green (G)  |   Blue (B)  | Formatted RGB Triplet\n");
        printf("-------+--------------+-------------+-------------+-------------+----------------------\n");
        for (int i = 0; i < num_pixels_to_show; i++) {
            int x = i % img->width;
            int y = i / img->width;
            if (img->bytes_per_sample == 1) {
                unsigned char r = img->raw_data[3 * i + 0];
                unsigned char g = img->raw_data[3 * i + 1];
                unsigned char b = img->raw_data[3 * i + 2];
                printf("  #%02d  |  (%4d, %4d) |     %3d     |     %3d     |     %3d     | RGB(%3d, %3d, %3d)\n",
                       i, x, y, r, g, b, r, g, b);
            } else {
                /* 16-bit PPM: 6 bytes per pixel (2 bytes R, 2 bytes G, 2 bytes B in Big-Endian) */
                unsigned short r = (unsigned short)((img->raw_data[6 * i + 0] << 8) | img->raw_data[6 * i + 1]);
                unsigned short g = (unsigned short)((img->raw_data[6 * i + 2] << 8) | img->raw_data[6 * i + 3]);
                unsigned short b = (unsigned short)((img->raw_data[6 * i + 4] << 8) | img->raw_data[6 * i + 5]);
                printf("  #%02d  |  (%4d, %4d) |    %5d    |    %5d    |    %5d    | RGB(%5d, %5d, %5d)\n",
                       i, x, y, r, g, b, r, g, b);
            }
        }
    }

    printf("\n  [16-Bit Sample Representation Explanation (MaxVal > 255)]:\n");
    printf("    * When MaxVal > 255 (e.g. MaxVal = 65535), each intensity sample requires 16 bits (2 bytes).\n");
    printf("    * Netpbm specifies Network Byte Order (Big-Endian):\n");
    printf("          Sample_Value = (Most_Significant_Byte << 8) | Least_Significant_Byte\n");
    printf("    * For PPM, each pixel consists of 3 consecutive 16-bit samples (6 bytes total: R_msb, R_lsb, G_msb, G_lsb, B_msb, B_lsb).\n");
    printf("========================================================================================\n");
}

/**
 * Extension vs Format Investigation
 */
void perform_extension_vs_format_investigation(const char *pgm_path, const char *ppm_path) {
    printf("\n\n################################################################################\n");
    printf("        INVESTIGATION: FILE EXTENSION VS FILE HEADER GROUND TRUTH               \n");
    printf("################################################################################\n");

    NetpbmImage *img_pgm = read_netpbm_image(pgm_path);
    NetpbmImage *img_ppm = read_netpbm_image(ppm_path);

    printf("\n========================================================================================\n");
    printf("                       EXTENSION VS HEADER FORMAT COMPARISON TABLE                      \n");
    printf("========================================================================================\n");
    printf(" Test Case Image | File Extension | Header Magic | Identified Format   | Channels / Pixel\n");
    printf("-----------------+----------------+--------------+---------------------+-----------------\n");
    if (img_pgm) {
        const char *ext = strrchr(img_pgm->filename, '.');
        printf(" File 1 (PGM)    | %-14s | %-12s | %-19s | %d Channel (Grayscale)\n",
               ext ? ext : "None", img_pgm->magic, "PGM Binary Raster", img_pgm->channels);
    }
    if (img_ppm) {
        const char *ext = strrchr(img_ppm->filename, '.');
        printf(" File 2 (PPM)    | %-14s | %-12s | %-19s | %d Channels (RGB Color)\n",
               ext ? ext : "None", img_ppm->magic, "PPM Binary Raster", img_ppm->channels);
    }
    printf("========================================================================================\n");

    printf("\n[Comprehensive Justification & Technical Answer]:\n");
    printf("  Question: Is the file extension alone sufficient to identify the image format?\n");
    printf("  Answer  : NO. The file extension is NOT sufficient.\n\n");
    printf("  Technical Justification:\n");
    printf("    1. Extensions are Superficial OS Conventions:\n");
    printf("       The filename extension (.pgm, .ppm, .bin) is merely an operating system naming convention\n");
    printf("       used for shell icon associations. A user or program can freely rename 'image.ppm' to 'image.pgm'\n");
    printf("       or 'image.dat' without altering the internal byte structure of the file.\n");
    printf("    2. The Header Magic Number is Authoritative Ground Truth:\n");
    printf("       Computer Vision algorithms and robust image loaders MUST read the initial bytes (magic number)\n");
    printf("       of the file stream. The magic number ('P5', 'P6', 'P2', 'P3') definitively dictates:\n");
    printf("         - Whether the raster is Grayscale (1 channel) or Truecolor (3 channels).\n");
    printf("         - Whether encoding is raw binary or ASCII text.\n");
    printf("         - How many bytes per pixel must be read to prevent memory corruption.\n");
    printf("========================================================================================\n");

    if (img_pgm) free_netpbm_image(img_pgm);
    if (img_ppm) free_netpbm_image(img_ppm);
}

/**
 * Critical-Thinking Component:
 * Consider an image with width = 640, height = 480 and MaxVal = 255.
 * Calculate expected pixel-data size for (i) PGM and (ii) PPM.
 * Repeat for MaxVal = 65535.
 * Explain why the byte count changes even though dimensions remain the same.
 */
void perform_critical_thinking_analysis(void) {
    int width = 640;
    int height = 480;
    long long total_pixels = (long long)width * height; /* 307,200 pixels */

    /* Case (i): MaxVal = 255 (8-bit depth) */
    int bytes_per_sample_8bit = 1;
    long long pgm_size_8bit = total_pixels * 1 * bytes_per_sample_8bit;
    long long ppm_size_8bit = total_pixels * 3 * bytes_per_sample_8bit;

    /* Case (ii): MaxVal = 65535 (16-bit depth) */
    int bytes_per_sample_16bit = 2;
    long long pgm_size_16bit = total_pixels * 1 * bytes_per_sample_16bit;
    long long ppm_size_16bit = total_pixels * 3 * bytes_per_sample_16bit;

    printf("\n\n################################################################################\n");
    printf("                    CRITICAL-THINKING ANALYTICAL COMPONENT                      \n");
    printf("################################################################################\n");
    printf(" Base Parameters: Image Dimensions = %d x %d | Total Pixels (N) = %lld pixels\n",
           width, height, total_pixels);

    printf("\n====================================================================================================\n");
    printf(" Case (i): Standard Dynamic Range (MaxVal = 255 -> 8-Bit Quantization Depth, 1 Byte/Sample)\n");
    printf("====================================================================================================\n");
    printf("  (a) PGM Format (Grayscale, 1 Channel):\n");
    printf("      * Formula       : Expected Bytes = Width * Height * Channels * BytesPerSample\n");
    printf("      * Calculation   : %d * %d * 1 * %d byte\n", width, height, bytes_per_sample_8bit);
    printf("      * Payload Size  : %lld bytes (%.2f KB / %.4f MB)\n",
           pgm_size_8bit, (double)pgm_size_8bit / 1024.0, (double)pgm_size_8bit / (1024.0 * 1024.0));
    printf("  (b) PPM Format (Truecolor RGB, 3 Channels):\n");
    printf("      * Formula       : Expected Bytes = Width * Height * Channels * BytesPerSample\n");
    printf("      * Calculation   : %d * %d * 3 * %d byte\n", width, height, bytes_per_sample_8bit);
    printf("      * Payload Size  : %lld bytes (%.2f KB / %.4f MB)\n",
           ppm_size_8bit, (double)ppm_size_8bit / 1024.0, (double)ppm_size_8bit / (1024.0 * 1024.0));

    printf("\n====================================================================================================\n");
    printf(" Case (ii): High Dynamic Range (MaxVal = 65535 -> 16-Bit Quantization Depth, 2 Bytes/Sample)\n");
    printf("====================================================================================================\n");
    printf("  (a) PGM Format (Grayscale, 1 Channel):\n");
    printf("      * Formula       : Expected Bytes = Width * Height * Channels * BytesPerSample\n");
    printf("      * Calculation   : %d * %d * 1 * %d bytes\n", width, height, bytes_per_sample_16bit);
    printf("      * Payload Size  : %lld bytes (%.2f KB / %.4f MB)\n",
           pgm_size_16bit, (double)pgm_size_16bit / 1024.0, (double)pgm_size_16bit / (1024.0 * 1024.0));
    printf("  (b) PPM Format (Truecolor RGB, 3 Channels):\n");
    printf("      * Formula       : Expected Bytes = Width * Height * Channels * BytesPerSample\n");
    printf("      * Calculation   : %d * %d * 3 * %d bytes\n", width, height, bytes_per_sample_16bit);
    printf("      * Payload Size  : %lld bytes (%.2f KB / %.4f MB)\n",
           ppm_size_16bit, (double)ppm_size_16bit / 1024.0, (double)ppm_size_16bit / (1024.0 * 1024.0));

    printf("\n====================================================================================================\n");
    printf(" Comparative Summary Table (640 x 480 Resolution):\n");
    printf("====================================================================================================\n");
    printf(" Format Type | MaxVal (Bit Depth) | Bytes/Pixel | Total Data Size (Bytes) | Size in KB | Size in MB\n");
    printf("-------------+--------------------+-------------+-------------------------+------------+-----------\n");
    printf(" PGM (Gray)  |   255  ( 8-bit)    |   1 byte    |        %7lld bytes     | %7.2f KB | %6.4f MB \n",
           pgm_size_8bit, (double)pgm_size_8bit / 1024.0, (double)pgm_size_8bit / (1024.0 * 1024.0));
    printf(" PPM (RGB)   |   255  ( 8-bit)    |   3 bytes   |        %7lld bytes     | %7.2f KB | %6.4f MB \n",
           ppm_size_8bit, (double)ppm_size_8bit / 1024.0, (double)ppm_size_8bit / (1024.0 * 1024.0));
    printf(" PGM (Gray)  | 65535  (16-bit)    |   2 bytes   |        %7lld bytes     | %7.2f KB | %6.4f MB \n",
           pgm_size_16bit, (double)pgm_size_16bit / 1024.0, (double)pgm_size_16bit / (1024.0 * 1024.0));
    printf(" PPM (RGB)   | 65535  (16-bit)    |   6 bytes   |        %7lld bytes     | %7.2f KB | %6.4f MB \n",
           ppm_size_16bit, (double)ppm_size_16bit / 1024.0, (double)ppm_size_16bit / (1024.0 * 1024.0));
    printf("====================================================================================================\n");

    printf("\n[Technical Explanation: Why the Number of Bytes Changes Despite Constant Dimensions]:\n");
    printf("  1. Spatial Resolution vs Radiometric (Intensity) Resolution:\n");
    printf("     - The spatial resolution (width x height = 640 x 480 = 307,200 spatial sample points) remains constant.\n");
    printf("     - However, the radiometric quantization depth (precision per sample point) is altered.\n\n");
    printf("  2. Information Theory & Bit-Depth Doubling:\n");
    printf("     - MaxVal = 255 requires 8 bits (1 byte) to represent 2^8 = 256 discrete gray/color levels.\n");
    printf("     - MaxVal = 65535 requires 16 bits (2 bytes) to represent 2^16 = 65,536 discrete intensity levels.\n");
    printf("     - Because every individual sample requires twice as many bits of storage, the total raster payload\n");
    printf("       DOUBLES exactly (100%% increase from 307.2 KB -> 614.4 KB for PGM, and 921.6 KB -> 1.84 MB for PPM).\n\n");
    printf("  3. Color Model Dimensionality Scaling:\n");
    printf("     - PPM stores three separate color components (Red, Green, Blue) per pixel.\n");
    printf("     - Hence, PPM is always exactly 3x the memory footprint of PGM at equivalent bit depths.\n");
    printf("====================================================================================================\n\n");
}

/**
 * Main application entry point
 */
int main(int argc, char *argv[]) {
    printf("================================================================================\n");
    printf("         NETPBM FORMAT PARSER, VERIFIER & CRITICAL ANALYSIS ENGINE (C)          \n");
    printf("================================================================================\n");

    char pgm_path_buf[512];
    char ppm_path_buf[512];

    const char *pgm_file = "image_1.pgm";
    const char *ppm_file = "image_1.ppm";

    if (argc >= 3) {
        pgm_file = argv[1];
        ppm_file = argv[2];
    }

    const char *resolved_pgm = resolve_input_path(pgm_file, pgm_path_buf, sizeof(pgm_path_buf));
    const char *resolved_ppm = resolve_input_path(ppm_file, ppm_path_buf, sizeof(ppm_path_buf));

    /* ============================================================================
     * PART 1: READ, ANALYZE, CALCULATE, AND VERIFY PGM IMAGE
     * ============================================================================ */
    printf("\n################################################################################\n");
    printf("        SECTION 1: PROCESSING PGM GRAYSCALE IMAGE (%s)                          \n", resolved_pgm);
    printf("################################################################################\n");
    NetpbmImage *pgm_img = read_netpbm_image(resolved_pgm);
    if (pgm_img) {
        display_complete_image_report(pgm_img);
        display_first_ten_pixels(pgm_img);
    } else {
        fprintf(stderr, "[FAILED] Could not load PGM image '%s'\n", resolved_pgm);
    }

    /* ============================================================================
     * PART 2: READ, ANALYZE, CALCULATE, AND VERIFY PPM IMAGE
     * ============================================================================ */
    printf("\n################################################################################\n");
    printf("        SECTION 2: PROCESSING PPM TRUECOLOR RGB IMAGE (%s)                      \n", resolved_ppm);
    printf("################################################################################\n");
    NetpbmImage *ppm_img = read_netpbm_image(resolved_ppm);
    if (ppm_img) {
        display_complete_image_report(ppm_img);
        display_first_ten_pixels(ppm_img);
    } else {
        fprintf(stderr, "[FAILED] Could not load PPM image '%s'\n", resolved_ppm);
    }

    /* ============================================================================
     * PART 3: EXTENSION VS FORMAT INVESTIGATION
     * ============================================================================ */
    perform_extension_vs_format_investigation(resolved_pgm, resolved_ppm);

    /* ============================================================================
     * PART 4: CRITICAL-THINKING ANALYTICAL COMPONENT (640x480 MaxVal=255 vs 65535)
     * ============================================================================ */
    perform_critical_thinking_analysis();

    /* Clean up allocated resources */
    if (pgm_img) free_netpbm_image(pgm_img);
    if (ppm_img) free_netpbm_image(ppm_img);

    printf("\n[SUCCESS] Lab 2 Netpbm identification, parsing, verification, and critical analysis completed.\n");
    return 0;
}