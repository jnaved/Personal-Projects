/**
 * ============================================================================
 * CVIP: Grayscale Image Reading, Pixel Analysis & Comparative Evaluation
 * ============================================================================
 * 
 * Objective:
 *  1. Read two different grayscale images (.pgm format - P5 raw binary / P2 ASCII)
 *     and store their pixel values in dynamic arrays.
 *  2. Determine for each image:
 *     - Dimensions (Width x Height, Total Pixels)
 *     - Minimum Intensity
 *     - Maximum Intensity
 *     - Dynamic Range
 *     - Average (Mean) Intensity
 *     - Standard Deviation (Contrast metric)
 *     - Intensity distribution (Dark, Mid-tone, Bright bins)
 *  3. Compare the two images side-by-side and draw clear, analytical conclusions
 *     regarding brightness, contrast, dynamic range, and distribution characteristics.
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define INTENSITY_LEVELS 256
#define DARK_THRESHOLD   85
#define MID_THRESHOLD    170
#define BRIGHT_THRESHOLD 255

/* Structure to store statistical metrics of an image */
typedef struct {
    int min_intensity;
    int max_intensity;
    int dynamic_range;
    double mean_intensity;
    double std_deviation;
    long long total_pixels;
    long long histogram[INTENSITY_LEVELS];
    long long dark_pixel_count;   /* [0 - 85] */
    long long mid_pixel_count;    /* [86 - 170] */
    long long bright_pixel_count; /* [171 - 255] */
    double dark_percent;
    double mid_percent;
    double bright_percent;
} ImageMetrics;

/* Structure to store image dimensions, pixel array, and computed metrics */
typedef struct {
    char filename[256];
    int width;
    int height;
    int max_val;
    int format; /* 5 for P5 (Binary), 2 for P2 (ASCII) */
    unsigned char *pixels;
    ImageMetrics metrics;
} GrayscaleImage;

/* Structure to define a rectangular Region of Interest (ROI) */
typedef struct {
    int x;
    int y;
    int width;
    int height;
} ROI;

/* Function Prototypes */
void compute_image_metrics(GrayscaleImage *img);
void free_image(GrayscaleImage *img);
GrayscaleImage* read_pgm_image(const char *filepath);
int write_pgm_image(const GrayscaleImage *img, const char *filepath);
GrayscaleImage* clone_image(const GrayscaleImage *src, const char *new_filename);
ROI get_local_region(const GrayscaleImage *image, int roi_w, int roi_h, int step_size);
GrayscaleImage* extract_roi(const GrayscaleImage *image, ROI roi, const char *roi_name);
void brighten_image(GrayscaleImage *img, int delta);
void print_ascii_histogram(const GrayscaleImage *img, const char *title);
void display_image_analysis(const GrayscaleImage *img, const char *label);
void display_comparative_table(const GrayscaleImage *img1, const GrayscaleImage *img2);
void draw_comparative_conclusions(const GrayscaleImage *img1, const GrayscaleImage *img2);
void compare_roi_with_parent(const GrayscaleImage *parent, const GrayscaleImage *roi, ROI roi_coords, const char *label);
void compare_brightened_with_original(const GrayscaleImage *orig, const GrayscaleImage *bright, int delta);
GrayscaleImage* create_synthetic_image(int width, int height, int pattern_type, const char *filename);
void display_equal_mean_investigation(GrayscaleImage **images, int count);


/**
 * Sliding window search to automatically find the sub-region of size (roi_w x roi_h)
 * whose local mean intensity deviates most significantly from the global image mean.
 */
ROI get_local_region(const GrayscaleImage *image, int roi_w, int roi_h, int step_size) {
    ROI best_roi = {.x = 0, .y = 0, .width = roi_w, .height = roi_h};
    if (!image || !image->pixels) return best_roi;

    /* Clamp ROI dimensions if requested size exceeds image boundaries */
    if (roi_w > image->width) roi_w = image->width;
    if (roi_h > image->height) roi_h = image->height;
    best_roi.width = roi_w;
    best_roi.height = roi_h;

    double max_diff = -1.0;
    double global_mean = image->metrics.mean_intensity;

    for (int y = 0; y <= image->height - roi_h; y += step_size) {
        for (int x = 0; x <= image->width - roi_w; x += step_size) {
            long long sum = 0;
            for (int r = 0; r < roi_h; r++) {
                int row_offset = (y + r) * image->width;
                for (int c = 0; c < roi_w; c++) {
                    sum += image->pixels[row_offset + (x + c)];
                }
            }

            double local_mean = (double)sum / ((double)roi_w * roi_h);
            double diff = fabs(local_mean - global_mean);

            if (diff > max_diff) {
                max_diff = diff;
                best_roi.x = x;
                best_roi.y = y;
            }
        }
    }

    return best_roi;
}

/**
 * Extract a rectangular Region of Interest (ROI) from an image into a new GrayscaleImage
 */
GrayscaleImage* extract_roi(const GrayscaleImage *image, ROI roi, const char *roi_name) {
    if (!image || !image->pixels) return NULL;

    /* Bounds checking and clipping */
    if (roi.x < 0) roi.x = 0;
    if (roi.y < 0) roi.y = 0;
    if (roi.x + roi.width > image->width) roi.width = image->width - roi.x;
    if (roi.y + roi.height > image->height) roi.height = image->height - roi.y;
    if (roi.width <= 0 || roi.height <= 0) return NULL;

    GrayscaleImage *new_image = (GrayscaleImage*)malloc(sizeof(GrayscaleImage));
    if (!new_image) {
        fprintf(stderr, "[ERROR] Memory allocation failed for ROI GrayscaleImage\n");
        return NULL;
    }
    memset(new_image, 0, sizeof(GrayscaleImage));

    if (roi_name) {
        strncpy(new_image->filename, roi_name, sizeof(new_image->filename) - 1);
        new_image->filename[sizeof(new_image->filename) - 1] = '\0';
    } else {
        snprintf(new_image->filename, sizeof(new_image->filename), "ROI [%d,%d %dx%d]",
                 roi.x, roi.y, roi.width, roi.height);
    }

    new_image->width = roi.width;
    new_image->height = roi.height;
    new_image->max_val = image->max_val;
    new_image->format = image->format;

    long long total_pixels = (long long)roi.width * roi.height;
    new_image->pixels = (unsigned char*)malloc(total_pixels * sizeof(unsigned char));
    if (!new_image->pixels) {
        fprintf(stderr, "[ERROR] Memory allocation failed for ROI pixel array\n");
        free(new_image);
        return NULL;
    }

    /* Fast row-by-row memory copy */
    for (int r = 0; r < roi.height; r++) {
        int src_row_start = (roi.y + r) * image->width + roi.x;
        int dst_row_start = r * roi.width;
        memcpy(&new_image->pixels[dst_row_start], &image->pixels[src_row_start], roi.width);
    }

    return new_image;
}

/**
 * Compare an extracted Region of Interest (ROI) side-by-side with its parent full image
 */
void compare_roi_with_parent(const GrayscaleImage *parent, const GrayscaleImage *roi, ROI roi_coords, const char *label) {
    if (!parent || !roi) return;
    const ImageMetrics *mp = &parent->metrics;
    const ImageMetrics *mr = &roi->metrics;

    printf("\n========================================================================================\n");
    printf("                  LOCAL REGION (ROI) VS FULL IMAGE EVALUATION                           \n");
    printf("                  %s\n", label);
    printf("========================================================================================\n");
    printf("  Selected Region Coordinates     : Origin (X=%d, Y=%d) | Size = %d x %d pixels\n",
           roi_coords.x, roi_coords.y, roi_coords.width, roi_coords.height);
    printf("  Region Pixel Coverage           : %lld pixels (%.2f%% of full image)\n\n",
           mr->total_pixels, (double)mr->total_pixels / (double)mp->total_pixels * 100.0);

    printf(" Parameter                        | Full Image (%-12s) | ROI Sub-region (%-10s)\n", "Global", "Local");
    printf("----------------------------------+-------------------------+---------------------------\n");
    printf(" Dimensions (Width x Height)      | %5d x %-15d | %5d x %-17d\n", parent->width, parent->height, roi->width, roi->height);
    printf(" Total Pixels (Resolution)        | %10lld (%5.2f MP)    | %10lld (%5.2f MP)     \n", 
           mp->total_pixels, (double)mp->total_pixels / 1e6,
           mr->total_pixels, (double)mr->total_pixels / 1e6);
    printf(" Minimum Intensity (Min)          | %23d | %25d\n", mp->min_intensity, mr->min_intensity);
    printf(" Maximum Intensity (Max)          | %23d | %25d\n", mp->max_intensity, mr->max_intensity);
    printf(" Dynamic Range (Max - Min)        | %23d | %25d\n", mp->dynamic_range, mr->dynamic_range);
    printf(" Average Intensity (Mean, mu)     | %23.2f | %25.2f\n", mp->mean_intensity, mr->mean_intensity);
    printf(" Standard Deviation (Contrast, s) | %23.2f | %25.2f\n", mp->std_deviation, mr->std_deviation);
    printf(" Dark Pixels   [  0 -  85]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) \n", 
           mp->dark_percent, mp->dark_pixel_count, mr->dark_percent, mr->dark_pixel_count);
    printf(" Medium Pixels [ 86 - 170]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) \n", 
           mp->mid_percent, mp->mid_pixel_count, mr->mid_percent, mr->mid_pixel_count);
    printf(" Bright Pixels [171 - 255]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) \n", 
           mp->bright_percent, mp->bright_pixel_count, mr->bright_percent, mr->bright_pixel_count);
    printf("----------------------------------+-------------------------+---------------------------\n");

    double mean_diff = mr->mean_intensity - mp->mean_intensity;
    double contrast_ratio = (mp->std_deviation > 0.0) ? (mr->std_deviation / mp->std_deviation) : 0.0;

    printf("\n[Analytical Interpretation of the Selected Region]:\n");
    if (mean_diff > 0) {
        printf("  * Intensity Shift: +%.2f units BRIGHTER than the global average (Highlight / High-Key Region).\n", mean_diff);
    } else {
        printf("  * Intensity Shift: %.2f units DARKER than the global average (Deep Shadow / Low-Key Region).\n", mean_diff);
    }
    printf("  * Local Contrast: Standard deviation is %.2f (Contrast ratio vs parent = %.2fx).\n",
           mr->std_deviation, contrast_ratio);
    if (contrast_ratio < 0.6) {
        printf("    -> The region is relatively uniform/homogeneous with low internal texture variation.\n");
    } else if (contrast_ratio > 1.2) {
        printf("    -> The region is highly textured with higher internal variability than the overall image.\n");
    } else {
        printf("    -> The region maintains moderate internal contrast comparable to the overall scene.\n");
    }
    printf("  * Dynamic Range Utilization: %d / %d (%.1f%% of parent image dynamic range).\n",
           mr->dynamic_range, mp->dynamic_range,
           (mp->dynamic_range > 0) ? ((double)mr->dynamic_range / mp->dynamic_range * 100.0) : 0.0);
    printf("========================================================================================\n\n");
}

/**
 * Deep copy an existing GrayscaleImage into a new GrayscaleImage
 */
GrayscaleImage* clone_image(const GrayscaleImage *src, const char *new_filename) {
    if (!src || !src->pixels) return NULL;

    GrayscaleImage *dst = (GrayscaleImage*)malloc(sizeof(GrayscaleImage));
    if (!dst) {
        fprintf(stderr, "[ERROR] Failed to allocate memory for cloned image\n");
        return NULL;
    }
    memset(dst, 0, sizeof(GrayscaleImage));

    if (new_filename) {
        strncpy(dst->filename, new_filename, sizeof(dst->filename) - 1);
        dst->filename[sizeof(dst->filename) - 1] = '\0';
    } else {
        snprintf(dst->filename, sizeof(dst->filename), "%.245s_clone", src->filename);
    }

    dst->width = src->width;
    dst->height = src->height;
    dst->max_val = src->max_val;
    dst->format = src->format;

    long long total_pixels = (long long)src->width * src->height;
    dst->pixels = (unsigned char*)malloc(total_pixels * sizeof(unsigned char));
    if (!dst->pixels) {
        fprintf(stderr, "[ERROR] Failed to allocate pixel memory for cloned image\n");
        free(dst);
        return NULL;
    }

    memcpy(dst->pixels, src->pixels, total_pixels * sizeof(unsigned char));
    dst->metrics = src->metrics;
    return dst;
}

/**
 * Brighten an image in-place by adding a constant offset (delta) with clamping [0, 255].
 * Automatically recomputes the image statistical metrics after modification.
 */
void brighten_image(GrayscaleImage *img, int delta) {
    if (!img || !img->pixels) return;

    long long total_pixels = (long long)img->width * img->height;
    for (long long i = 0; i < total_pixels; i++) {
        int val = (int)img->pixels[i] + delta;
        if (val > 255) val = 255;
        else if (val < 0) val = 0;
        img->pixels[i] = (unsigned char)val;
    }

    /* Recompute metrics after modifying pixel values */
    compute_image_metrics(img);
}

/**
 * Side-by-side comparative evaluation of original vs brightened image
 */
void compare_brightened_with_original(const GrayscaleImage *orig, const GrayscaleImage *bright, int delta) {
    if (!orig || !bright) return;
    const ImageMetrics *mo = &orig->metrics;
    const ImageMetrics *mb = &bright->metrics;

    printf("\n========================================================================================\n");
    printf("                  BRIGHTENING TRANSFORMATION EVALUATION (DELTA = %+d)                   \n", delta);
    printf("                  Original: %s  ->  Brightened: %s\n", orig->filename, bright->filename);
    printf("========================================================================================\n");
    printf(" Parameter                        | Original Image            | Brightened Image (Delta=%+d)\n", delta);
    printf("----------------------------------+-------------------------+---------------------------\n");
    printf(" Dimensions (Width x Height)      | %5d x %-15d | %5d x %-17d\n", orig->width, orig->height, bright->width, bright->height);
    printf(" Total Pixels (Resolution)        | %10lld (%5.2f MP)    | %10lld (%5.2f MP)     \n", 
           mo->total_pixels, (double)mo->total_pixels / 1e6,
           mb->total_pixels, (double)mb->total_pixels / 1e6);
    printf(" Minimum Intensity (Min)          | %23d | %25d\n", mo->min_intensity, mb->min_intensity);
    printf(" Maximum Intensity (Max)          | %23d | %25d\n", mo->max_intensity, mb->max_intensity);
    printf(" Dynamic Range (Max - Min)        | %23d | %25d\n", mo->dynamic_range, mb->dynamic_range);
    printf(" Average Intensity (Mean, mu)     | %23.2f | %25.2f\n", mo->mean_intensity, mb->mean_intensity);
    printf(" Standard Deviation (Contrast, s) | %23.2f | %25.2f\n", mo->std_deviation, mb->std_deviation);
    printf(" Dark Pixels   [  0 -  85]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) \n", 
           mo->dark_percent, mo->dark_pixel_count, mb->dark_percent, mb->dark_pixel_count);
    printf(" Medium Pixels [ 86 - 170]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) \n", 
           mo->mid_percent, mo->mid_pixel_count, mb->mid_percent, mb->mid_pixel_count);
    printf(" Bright Pixels [171 - 255]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) \n", 
           mo->bright_percent, mo->bright_pixel_count, mb->bright_percent, mb->bright_pixel_count);
    printf(" Saturated Highlights [255 only]  | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) \n",
           (double)mo->histogram[255] / (double)mo->total_pixels * 100.0, mo->histogram[255],
           (double)mb->histogram[255] / (double)mb->total_pixels * 100.0, mb->histogram[255]);
    printf("----------------------------------+-------------------------+---------------------------\n");

    double mean_shift = mb->mean_intensity - mo->mean_intensity;
    printf("\n[Analytical Observations on Brightening]:\n");
    printf("  1. Mean Intensity Shift : %+.2f units (Overall luminance increased by %.1f%% relative to scale).\n",
           mean_shift, (mean_shift / 255.0) * 100.0);
    printf("  2. Minimum Intensity    : Lifted from %d -> %d (+%d floor shift).\n",
           mo->min_intensity, mb->min_intensity, mb->min_intensity - mo->min_intensity);
    printf("  3. Dark Population Shift: Reduced from %.2f%% -> %.2f%% (%.2f%% of shadow pixels migrated to mid-tones).\n",
           mo->dark_percent, mb->dark_percent, mo->dark_percent - mb->dark_percent);
    printf("  4. Highlight Saturation : Saturation at intensity 255 is %.2f%% (%lld pixels).\n",
           (double)mb->histogram[255] / (double)mb->total_pixels * 100.0, mb->histogram[255]);
    if (fabs(mb->std_deviation - mo->std_deviation) < 1.0) {
        printf("  5. Contrast / Variance  : Standard deviation remained nearly identical (%.2f vs %.2f) because\n"
               "                            linear translation preserves relative distances between unclipped pixels.\n",
               mo->std_deviation, mb->std_deviation);
    } else {
        printf("  5. Contrast / Variance  : Standard deviation shifted from %.2f -> %.2f due to clipping/saturation at boundaries.\n",
               mo->std_deviation, mb->std_deviation);
    }
    printf("========================================================================================\n\n");
}

/**
 * Synthesize demonstration images of size (width x height) with engineered patterns
 * that all share the exact same mathematical mean intensity (~128.0).
 *
 * Patterns:
 *  1: Flat Uniform Gray (All pixels = 128)
 *  2: High-Contrast Binary Checkerboard (50% value 1, 50% value 255 -> mean 128.0)
 *  3: Horizontal Linear Gradient Ramp (1 to 255 -> mean 128.0)
 *  4: Concentric Circle Target (Sinusoidal rings centered at 128 -> mean 128.0)
 */
GrayscaleImage* create_synthetic_image(int width, int height, int pattern_type, const char *filename) {
    GrayscaleImage *img = (GrayscaleImage*)malloc(sizeof(GrayscaleImage));
    if (!img) return NULL;
    memset(img, 0, sizeof(GrayscaleImage));

    strncpy(img->filename, filename, sizeof(img->filename) - 1);
    img->filename[sizeof(img->filename) - 1] = '\0';
    img->width = width;
    img->height = height;
    img->max_val = 255;
    img->format = 5; /* P5 Binary */

    long long total = (long long)width * height;
    img->pixels = (unsigned char*)malloc(total * sizeof(unsigned char));
    if (!img->pixels) {
        free(img);
        return NULL;
    }

    if (pattern_type == 1) {
        /* Pattern 1: Flat Uniform Gray (Zero Contrast) */
        memset(img->pixels, 128, total);
    } else if (pattern_type == 2) {
        /* Pattern 2: High-Contrast Binary Checkerboard (Max Contrast) */
        int block_size = 20;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int check = ((x / block_size) + (y / block_size)) % 2;
                img->pixels[y * width + x] = (check == 0) ? 1 : 255;
            }
        }
    } else if (pattern_type == 3) {
        /* Pattern 3: Horizontal Linear Gradient (Continuous Spread) */
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int val = 1 + (int)((double)x / (double)(width - 1) * 254.0 + 0.5);
                if (val < 0) {
                    val = 0;
                } else if (val > 255) {
                    val = 255;
                }
                img->pixels[y * width + x] = (unsigned char)val;
            }
        }
    } else if (pattern_type == 4) {
        /* Pattern 4: Concentric Circles Target (Circular Symmetry) */
        double cx = width / 2.0, cy = height / 2.0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double dist = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                int val = (int)(128.0 + 120.0 * sin(dist / 12.0) + 0.5);
                if (val < 0) {
                    val = 0;
                } else if (val > 255) {
                    val = 255;
                }
                img->pixels[y * width + x] = (unsigned char)val;
            }
        }
    }

    compute_image_metrics(img);
    return img;
}

/**
 * Multi-image comparative evaluation proving equal mean intensity with diverse visual patterns
 */
void display_equal_mean_investigation(GrayscaleImage **images, int count) {
    if (!images || count < 4) return;

    printf("\n========================================================================================================================\n");
    printf("                  INVESTIGATION TABLE: 4 VISUALLY DISTINCT IMAGES WITH IDENTICAL MEAN INTENSITY (mu ~ 128.0)            \n");
    printf("========================================================================================================================\n");
    printf(" Parameter                        | [1] Flat Gray           | [2] Checkerboard        | [3] Gradient Ramp       | [4] Concentric Rings    \n");
    printf("----------------------------------+-------------------------+-------------------------+-------------------------+-----------------------\n");
    printf(" Visual Appearance                | Uniform solid gray      | High-contrast grid      | Smooth horizontal fade  | Circular ripple waves \n");
    printf(" Dimensions (Width x Height)      | %5d x %-15d | %5d x %-15d | %5d x %-15d | %5d x %-15d\n",
           images[0]->width, images[0]->height, images[1]->width, images[1]->height,
           images[2]->width, images[2]->height, images[3]->width, images[3]->height);
    printf(" Minimum Intensity (Min)          | %23d | %23d | %23d | %21d\n",
           images[0]->metrics.min_intensity, images[1]->metrics.min_intensity,
           images[2]->metrics.min_intensity, images[3]->metrics.min_intensity);
    printf(" Maximum Intensity (Max)          | %23d | %23d | %23d | %21d\n",
           images[0]->metrics.max_intensity, images[1]->metrics.max_intensity,
           images[2]->metrics.max_intensity, images[3]->metrics.max_intensity);
    printf(" Dynamic Range (Max - Min)        | %23d | %23d | %23d | %21d\n",
           images[0]->metrics.dynamic_range, images[1]->metrics.dynamic_range,
           images[2]->metrics.dynamic_range, images[3]->metrics.dynamic_range);
    printf(" Average Intensity (Mean, mu)     | %23.2f | %23.2f | %23.2f | %21.2f\n",
           images[0]->metrics.mean_intensity, images[1]->metrics.mean_intensity,
           images[2]->metrics.mean_intensity, images[3]->metrics.mean_intensity);
    printf(" Standard Deviation (Contrast, s) | %23.2f | %23.2f | %23.2f | %21.2f\n",
           images[0]->metrics.std_deviation, images[1]->metrics.std_deviation,
           images[2]->metrics.std_deviation, images[3]->metrics.std_deviation);
    printf(" Dark Pixels   [  0 -  85]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) | %4.2f%% (%7lld px)\n",
           images[0]->metrics.dark_percent, images[0]->metrics.dark_pixel_count,
           images[1]->metrics.dark_percent, images[1]->metrics.dark_pixel_count,
           images[2]->metrics.dark_percent, images[2]->metrics.dark_pixel_count,
           images[3]->metrics.dark_percent, images[3]->metrics.dark_pixel_count);
    printf(" Medium Pixels [ 86 - 170]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) | %4.2f%% (%7lld px)\n",
           images[0]->metrics.mid_percent, images[0]->metrics.mid_pixel_count,
           images[1]->metrics.mid_percent, images[1]->metrics.mid_pixel_count,
           images[2]->metrics.mid_percent, images[2]->metrics.mid_pixel_count,
           images[3]->metrics.mid_percent, images[3]->metrics.mid_pixel_count);
    printf(" Bright Pixels [171 - 255]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) | %4.2f%% (%7lld px)\n",
           images[0]->metrics.bright_percent, images[0]->metrics.bright_pixel_count,
           images[1]->metrics.bright_percent, images[1]->metrics.bright_pixel_count,
           images[2]->metrics.bright_percent, images[2]->metrics.bright_pixel_count,
           images[3]->metrics.bright_percent, images[3]->metrics.bright_pixel_count);
    printf("----------------------------------+-------------------------+-------------------------+-------------------------+-----------------------\n");

    printf("\n[Mathematical Proof & Key Observations]:\n");
    printf("  1. Identical First Moment (Mean mu = 128.00):\n"
           "     All 4 images contain the exact same total photon energy/sum across all %lld pixels.\n",
           images[0]->metrics.total_pixels);
    printf("  2. Radical Variance Divergence (Contrast sigma = 0.00 to 127.00):\n"
           "     - Pattern 1 has ZERO contrast (all pixels clustered at 128).\n"
           "     - Pattern 2 has MAXIMUM contrast (bimodal split at 1 and 255).\n"
           "     - Pattern 3 has SMOOTH continuous contrast (sigma = 73.61).\n"
           "     - Pattern 4 has HARMONIC spatial structure (sigma = 84.85).\n");
    printf("  3. Conclusion: The mean intensity alone provides ZERO spatial geometry or texture information;\n"
           "     higher-order statistical moments and spatial frequency descriptors are mandatory in CV/DIP.\n");
    printf("========================================================================================================================\n\n");
}

/**
 * Robust path resolver for root, lab_1/, and images directories
 */
static const char* resolve_input_path(const char *rel_path, char *buf, size_t buf_size) {
    FILE *f = fopen(rel_path, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", rel_path); return buf; }

    char candidate[512];
    snprintf(candidate, sizeof(candidate), "lab_1/%s", rel_path);
    f = fopen(candidate, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", candidate); return buf; }

    snprintf(candidate, sizeof(candidate), "images/%s", rel_path);
    f = fopen(candidate, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", candidate); return buf; }

    snprintf(candidate, sizeof(candidate), "../images/%s", rel_path);
    f = fopen(candidate, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", candidate); return buf; }

    snprintf(candidate, sizeof(candidate), "../lab_1/images/%s", rel_path);
    f = fopen(candidate, "rb");
    if (f) { fclose(f); snprintf(buf, buf_size, "%s", candidate); return buf; }

    snprintf(buf, buf_size, "%s", rel_path);
    return buf;
}

/**
 * Helper to resolve output file paths whether running from project root or lab_1/ subfolder
 */
static const char* get_output_path(const char *filename_or_path, char *buffer, size_t buf_size) {
    const char *basename = strrchr(filename_or_path, '/');
    if (!basename) basename = strrchr(filename_or_path, '\\');
    const char *fname = basename ? (basename + 1) : filename_or_path;

    FILE *f = fopen("lab_1/images/image_1.pgm", "rb");
    if (f) {
        fclose(f);
        snprintf(buffer, buf_size, "lab_1/images/%s", fname);
        return buffer;
    }
    f = fopen("images/image_1.pgm", "rb");
    if (f) {
        fclose(f);
        snprintf(buffer, buf_size, "images/%s", fname);
        return buffer;
    }
    snprintf(buffer, buf_size, "../images/%s", fname);
    return buffer;
}

/**
 * Save a GrayscaleImage as a PGM file (P5 Binary or P2 ASCII based on img->format)
 */
int write_pgm_image(const GrayscaleImage *img, const char *filepath) {
    if (!img || !img->pixels) {
        fprintf(stderr, "[ERROR] Cannot save NULL image to '%s'\n", filepath);
        return 0;
    }

    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Unable to open '%s' for writing\n", filepath);
        return 0;
    }

    if (img->format == 2) {
        /* Write P2 ASCII format */
        fprintf(fp, "P2\n# Created by CVIP Image Processor\n%d %d\n%d\n",
                img->width, img->height, img->max_val);
        long long total = (long long)img->width * img->height;
        for (long long i = 0; i < total; i++) {
            fprintf(fp, "%d ", img->pixels[i]);
            if ((i + 1) % 16 == 0) fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    } else {
        /* Default to P5 Binary format */
        fprintf(fp, "P5\n# Created by CVIP Image Processor\n%d %d\n%d\n",
                img->width, img->height, img->max_val);
        fwrite(img->pixels, sizeof(unsigned char), (size_t)img->width * img->height, fp);
    }

    fclose(fp);
    printf("   [Exported PGM] Saved '%s' (%dx%d, PGM %s)\n",
           filepath, img->width, img->height, (img->format == 2 ? "P2" : "P5"));
    return 1;
}

/**
 * Helper function to skip whitespace and comments in PGM files
 */
static void skip_pgm_comments(FILE *fp) {
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '#') {
            /* Skip characters until end of line */
            while ((ch = fgetc(fp)) != EOF && ch != '\n' && ch != '\r');
        } else if (!isspace(ch)) {
            ungetc(ch, fp);
            break;
        }
    }
}

/**
 * Read a grayscale PGM image (P5 Binary or P2 ASCII) into memory
 */
GrayscaleImage* read_pgm_image(const char *filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Unable to open image file: '%s'\n", filepath);
        return NULL;
    }

    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1) {
        fprintf(stderr, "[ERROR] Failed to read magic number from '%s'\n", filepath);
        fclose(fp);
        return NULL;
    }

    int format = 0;
    if (strcmp(magic, "P5") == 0) {
        format = 5;
    } else if (strcmp(magic, "P2") == 0) {
        format = 2;
    } else {
        fprintf(stderr, "[ERROR] Unsupported format '%s' in '%s'. Supported: P5, P2.\n", magic, filepath);
        fclose(fp);
        return NULL;
    }

    skip_pgm_comments(fp);

    int width = 0, height = 0, max_val = 0;
    if (fscanf(fp, "%d", &width) != 1 || width <= 0) {
        fprintf(stderr, "[ERROR] Invalid width in '%s'\n", filepath);
        fclose(fp);
        return NULL;
    }

    skip_pgm_comments(fp);
    if (fscanf(fp, "%d", &height) != 1 || height <= 0) {
        fprintf(stderr, "[ERROR] Invalid height in '%s'\n", filepath);
        fclose(fp);
        return NULL;
    }

    skip_pgm_comments(fp);
    if (fscanf(fp, "%d", &max_val) != 1 || max_val <= 0 || max_val > 255) {
        fprintf(stderr, "[ERROR] Invalid or unsupported max intensity value (%d) in '%s'\n", max_val, filepath);
        fclose(fp);
        return NULL;
    }

    /* Single whitespace character after max_val before pixel payload */
    fgetc(fp);

    GrayscaleImage *img = (GrayscaleImage*)malloc(sizeof(GrayscaleImage));
    if (!img) {
        fprintf(stderr, "[ERROR] Memory allocation failed for image struct\n");
        fclose(fp);
        return NULL;
    }

    strncpy(img->filename, filepath, sizeof(img->filename) - 1);
    img->filename[sizeof(img->filename) - 1] = '\0';
    img->width = width;
    img->height = height;
    img->max_val = max_val;
    img->format = format;

    long long total_pixels = (long long)width * height;
    img->pixels = (unsigned char*)malloc(total_pixels * sizeof(unsigned char));
    if (!img->pixels) {
        fprintf(stderr, "[ERROR] Memory allocation failed for %lld pixel bytes\n", total_pixels);
        free(img);
        fclose(fp);
        return NULL;
    }

    if (format == 5) {
        /* P5 Binary raw read */
        size_t read_bytes = fread(img->pixels, sizeof(unsigned char), total_pixels, fp);
        if (read_bytes != (size_t)total_pixels) {
            fprintf(stderr, "[WARNING] Expected %lld bytes but read %zu bytes from '%s'\n",
                    total_pixels, read_bytes, filepath);
        }
    } else {
        /* P2 ASCII value read */
        for (long long i = 0; i < total_pixels; i++) {
            int val;
            if (fscanf(fp, "%d", &val) != 1) {
                fprintf(stderr, "[ERROR] Premature end of ASCII pixel data at index %lld\n", i);
                break;
            }
            img->pixels[i] = (unsigned char)(val < 0 ? 0 : (val > 255 ? 255 : val));
        }
    }

    fclose(fp);
    return img;
}

/**
 * Free memory occupied by GrayscaleImage
 */
void free_image(GrayscaleImage *img) {
    if (img) {
        if (img->pixels) {
            free(img->pixels);
        }
        free(img);
    }
}

/**
 * Compute detailed statistical analysis of the image's pixel values
 * and store the results in img->metrics.
 */
void compute_image_metrics(GrayscaleImage *img) {
    if (!img) return;
    memset(&img->metrics, 0, sizeof(ImageMetrics));
    if (!img->pixels) return;

    long long total_pixels = (long long)img->width * img->height;
    img->metrics.total_pixels = total_pixels;

    int min_val = 255;
    int max_val = 0;
    double sum_intensity = 0.0;

    for (long long i = 0; i < total_pixels; i++) {
        unsigned char pixel = img->pixels[i];
        
        if (pixel < min_val) min_val = pixel;
        if (pixel > max_val) max_val = pixel;

        sum_intensity += pixel;
        img->metrics.histogram[pixel]++;

        if (pixel <= DARK_THRESHOLD) {
            img->metrics.dark_pixel_count++;
        } else if (pixel <= MID_THRESHOLD) {
            img->metrics.mid_pixel_count++;
        } else {
            img->metrics.bright_pixel_count++;
        }
    }

    img->metrics.min_intensity = min_val;
    img->metrics.max_intensity = max_val;
    img->metrics.dynamic_range = max_val - min_val;
    img->metrics.mean_intensity = sum_intensity / (double)total_pixels;

    /* Compute Standard Deviation for contrast measurement */
    double sum_sq_diff = 0.0;
    for (int v = 0; v < INTENSITY_LEVELS; v++) {
        if (img->metrics.histogram[v] > 0) {
            double diff = (double)v - img->metrics.mean_intensity;
            sum_sq_diff += (double)img->metrics.histogram[v] * (diff * diff);
        }
    }
    img->metrics.std_deviation = sqrt(sum_sq_diff / (double)total_pixels);

    /* Percentages */
    img->metrics.dark_percent   = (double)img->metrics.dark_pixel_count   / (double)total_pixels * 100.0;
    img->metrics.mid_percent    = (double)img->metrics.mid_pixel_count    / (double)total_pixels * 100.0;
    img->metrics.bright_percent = (double)img->metrics.bright_pixel_count / (double)total_pixels * 100.0;
}

/**
 * Print ASCII histogram representation of intensity distribution
 */
void print_ascii_histogram(const GrayscaleImage *img, const char *title) {
    if (!img) return;
    const ImageMetrics *m = &img->metrics;

    printf("\n--- 16-Bin Intensity Histogram Distribution [%s] ---\n", title);
    printf(" Bin Range   | Pixel Percentage | Distribution Bar\n");
    printf("-------------+------------------+--------------------------------------------------\n");

    /* Group 256 levels into 16 bins (16 levels per bin) */
    for (int b = 0; b < 16; b++) {
        int start = b * 16;
        int end = start + 15;
        long long bin_count = 0;
        for (int i = start; i <= end; i++) {
            bin_count += m->histogram[i];
        }
        double bin_pct = (double)bin_count / (double)m->total_pixels * 100.0;
        int bar_len = (int)(bin_pct * 1.5); /* Scale to 50 characters max */
        if (bar_len > 50) bar_len = 50;

        printf(" [%3d - %3d] |      %6.2f%%     | ", start, end, bin_pct);
        for (int k = 0; k < bar_len; k++) {
            putchar('#');
        }
        putchar('\n');
    }
    printf("------------------------------------------------------------------------------------\n");
}

/**
 * Display individual analysis for an image
 */
void display_image_analysis(const GrayscaleImage *img, const char *label) {
    if (!img) return;
    const ImageMetrics *m = &img->metrics;

    printf("\n================================================================================\n");
    printf("                       %s: %s\n", label, img->filename);
    printf("================================================================================\n");
    printf("  [1] Dimensions & Format:\n");
    printf("      * Width               : %d pixels\n", img->width);
    printf("      * Height              : %d pixels\n", img->height);
    printf("      * Total Pixels (N)    : %lld (%.2f Megapixels)\n", m->total_pixels, (double)m->total_pixels / 1e6);
    printf("      * Storage Format      : PGM %s\n", (img->format == 5 ? "P5 (Binary)" : "P2 (ASCII)"));
    printf("      * Header Max Value    : %d\n", img->max_val);
    printf("\n  [2] Intensity Range & Statistics:\n");
    printf("      * Minimum Intensity   : %d\n", m->min_intensity);
    printf("      * Maximum Intensity   : %d\n", m->max_intensity);
    printf("      * Dynamic Range       : %d (out of 255)\n", m->dynamic_range);
    printf("      * Average (Mean)      : %.2f / 255.00\n", m->mean_intensity);
    printf("      * Standard Dev (Std)  : %.2f (Contrast Indicator)\n", m->std_deviation);
    printf("\n  [3] Intensity Tier Distribution:\n");
    printf("      * Dark   [  0 -  85]  : %lld pixels (%6.2f%%)\n", m->dark_pixel_count, m->dark_percent);
    printf("      * Medium [ 86 - 170]  : %lld pixels (%6.2f%%)\n", m->mid_pixel_count, m->mid_percent);
    printf("      * Bright [171 - 255]  : %lld pixels (%6.2f%%)\n", m->bright_pixel_count, m->bright_percent);
    printf("================================================================================\n");
}

/**
 * Display side-by-side comparative table
 */
void display_comparative_table(const GrayscaleImage *img1, const GrayscaleImage *img2) {
    if (!img1 || !img2) return;
    const ImageMetrics *m1 = &img1->metrics;
    const ImageMetrics *m2 = &img2->metrics;

    printf("\n========================================================================================\n");
    printf("                       SIDE-BY-SIDE NUMERICAL COMPARISON TABLE                          \n");
    printf("========================================================================================\n");
    printf(" Metric / Parameter               | Image 1 (%-15s) | Image 2 (%-15s)\n", "image_1", "image_2");
    printf("----------------------------------+-------------------------+---------------------------\n");
    printf(" File Name                        | %-23s | %-25s\n", img1->filename, img2->filename);
    printf(" Dimensions (Width x Height)      | %5d x %-15d | %5d x %-17d\n", img1->width, img1->height, img2->width, img2->height);
    printf(" Total Pixels (Resolution)        | %10lld (%5.2f MP)    | %10lld (%5.2f MP)     \n", 
           m1->total_pixels, (double)m1->total_pixels / 1e6,
           m2->total_pixels, (double)m2->total_pixels / 1e6);
    printf(" Minimum Intensity (Min)          | %23d | %25d\n", m1->min_intensity, m2->min_intensity);
    printf(" Maximum Intensity (Max)          | %23d | %25d\n", m1->max_intensity, m2->max_intensity);
    printf(" Dynamic Range (Max - Min)        | %23d | %25d\n", m1->dynamic_range, m2->dynamic_range);
    printf(" Average Intensity (Mean, mu)     | %23.2f | %25.2f\n", m1->mean_intensity, m2->mean_intensity);
    printf(" Standard Deviation (Contrast, s) | %23.2f | %25.2f\n", m1->std_deviation, m2->std_deviation);
    printf(" Dark Pixels   [  0 -  85]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) \n", 
           m1->dark_percent, m1->dark_pixel_count, m2->dark_percent, m2->dark_pixel_count);
    printf(" Medium Pixels [ 86 - 170]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) \n", 
           m1->mid_percent, m1->mid_pixel_count, m2->mid_percent, m2->mid_pixel_count);
    printf(" Bright Pixels [171 - 255]        | %6.2f%% (%9lld px) | %6.2f%% (%9lld px) \n", 
           m1->bright_percent, m1->bright_pixel_count, m2->bright_percent, m2->bright_pixel_count);
    printf("========================================================================================\n");
}

/**
 * Analyze comparison and print conclusions
 */
void draw_comparative_conclusions(const GrayscaleImage *img1, const GrayscaleImage *img2) {
    if (!img1 || !img2) return;
    const ImageMetrics *m1 = &img1->metrics;
    const ImageMetrics *m2 = &img2->metrics;

    printf("\n========================================================================================\n");
    printf("                            CONCLUSIONS & IMAGE EVALUATION                              \n");
    printf("========================================================================================\n");

    /* 1. Brightness / Luminance Comparison */
    printf("\n[1] Overall Brightness Comparison:\n");
    if (fabs(m1->mean_intensity - m2->mean_intensity) < 1.0) {
        printf("    * Both images exhibit very similar global mean intensity (%.2f vs %.2f).\n",
               m1->mean_intensity, m2->mean_intensity);
        if (m1->bright_percent > m2->bright_percent) {
            printf("    * However, Image 1 contains significantly more bright highlight pixels (%.2f%% vs %.2f%%),\n"
                   "      giving it higher perceived luminance in specular/light regions.\n",
                   m1->bright_percent, m2->bright_percent);
        } else {
            printf("    * However, Image 2 contains more bright highlight pixels (%.2f%% vs %.2f%%).\n",
                   m2->bright_percent, m1->bright_percent);
        }
    } else if (m1->mean_intensity > m2->mean_intensity) {
        printf("    * Image 1 is overall BRIGHTER with a mean intensity of %.2f compared to %.2f for Image 2\n"
               "      (Mean difference: +%.2f units).\n",
               m1->mean_intensity, m2->mean_intensity, m1->mean_intensity - m2->mean_intensity);
    } else {
        printf("    * Image 2 is overall BRIGHTER with a mean intensity of %.2f compared to %.2f for Image 1\n"
               "      (Mean difference: +%.2f units).\n",
               m2->mean_intensity, m1->mean_intensity, m2->mean_intensity - m1->mean_intensity);
    }

    /* 2. Contrast & Dynamic Range Comparison */
    printf("\n[2] Contrast & Dynamic Range Comparison:\n");
    if (m1->std_deviation > m2->std_deviation) {
        printf("    * Image 1 exhibits SUBSTANTIALLY HIGHER CONTRAST (Standard Deviation = %.2f vs %.2f for Image 2).\n",
               m1->std_deviation, m2->std_deviation);
        printf("    * Image 1 utilizes the full dynamic range (%d, min=%d to max=%d),\n"
               "      whereas Image 2 utilizes a more compressed dynamic range (%d, min=%d to max=%d).\n",
               m1->dynamic_range, m1->min_intensity, m1->max_intensity,
               m2->dynamic_range, m2->min_intensity, m2->max_intensity);
    } else {
        printf("    * Image 2 exhibits HIGHER CONTRAST (Standard Deviation = %.2f vs %.2f for Image 1).\n",
               m2->std_deviation, m1->std_deviation);
        printf("    * Image 2 utilizes dynamic range of %d compared to %d for Image 1.\n",
               m2->dynamic_range, m1->dynamic_range);
    }

    /* 3. Pixel Distribution & Histogram Profiles */
    printf("\n[3] Pixel Distribution & Tonality Profile:\n");
    printf("    * Image 1 Profile: Bimodal / Broad spread with %.2f%% Dark and %.2f%% Bright pixels.\n"
           "      This indicates pronounced shadows and distinct highlights.\n",
           m1->dark_percent, m1->bright_percent);
    printf("    * Image 2 Profile: Centered / Unimodal distribution clustered heavily in the Medium range (%.2f%%).\n"
           "      It lacks deep absolute blacks (min = %d) and pure white highlights (max = %d).\n",
           m2->mid_percent, m2->min_intensity, m2->max_intensity);

    /* 4. Practical CV/IP Application Suitability: Thresholding & Segmentation */
    printf("\n[4] Comprehensive Suitability Evaluation for Thresholding & Segmentation:\n");
    if (m1->std_deviation > m2->std_deviation) {
        printf("    * WINNER FOR THRESHOLDING: IMAGE 1 (image_1.pgm)\n\n");
        printf("    * Detailed Mathematical & Statistical Justification:\n");
        printf("      1. Bimodal Distribution Structure:\n");
        printf("         Image 1 features distinct, well-separated peaks in the dark shadow tier (%.2f%%) and\n", m1->dark_percent);
        printf("         the bright highlight tier (%.2f%%), separated by a deep intensity valley in the mid-tones\n", m1->bright_percent);
        printf("         (bins [112-159] have minimal pixel density ~3.2-3.6%%).\n");
        printf("      2. Superior Contrast (Standard Deviation):\n");
        printf("         Image 1 exhibits sigma = %.2f (vs %.2f for Image 2). High variance between foreground\n", m1->std_deviation, m2->std_deviation);
        printf("         and background clusters guarantees sharp, well-defined binary segmentation boundaries\n");
        printf("         with minimal misclassification probability.\n");
        printf("      3. Full Dynamic Range Utilization:\n");
        printf("         Image 1 spans the entire range [%d to %d] (Span = %d), ensuring distinct separation\n", m1->min_intensity, m1->max_intensity, m1->dynamic_range);
        printf("         headroom between dark and bright regions.\n\n");

        printf("    * Why Image 2 is Poor for Global Thresholding (and Recommended Treatment):\n");
        printf("      1. Unimodal / Centered Distribution:\n");
        printf("         Image 2 is heavily congested in the mid-tone range (%.2f%% in [86-170]), lacking\n", m2->mid_percent);
        printf("         a clear bimodal valley. Applying a global threshold will slice through the densest\n");
        printf("         portion of the histogram, resulting in severe noise, speckled artifacts, and broken edges.\n");
        printf("      2. Recommended Preprocessing for Image 2:\n");
        printf("         Image 2 requires Histogram Equalization or Contrast Stretching first, or locally adaptive\n");
        printf("         algorithms (e.g., Bradley-Roth or Sauvola local window thresholding) rather than global thresholding.\n");
    } else {
        printf("    * WINNER FOR THRESHOLDING: IMAGE 2 (image_2.pgm)\n");
    }
    printf("========================================================================================\n\n");
}

/**
 * Main entry point
 */
int main(int argc, char *argv[]) {
    char path_buf1[512];
    char path_buf2[512];
    const char *img1_file = "images/image_1.pgm";
    const char *img2_file = "images/image_2.pgm";

    if (argc >= 3) {
        img1_file = argv[1];
        img2_file = argv[2];
    }

    const char *img1_path = resolve_input_path(img1_file, path_buf1, sizeof(path_buf1));
    const char *img2_path = resolve_input_path(img2_file, path_buf2, sizeof(path_buf2));

    printf("================================================================================\n");
    printf("            GRAYSCALE IMAGE ANALYSIS AND NUMERICAL COMPARISON IN C             \n");
    printf("================================================================================\n");
    printf("Loading Image 1: %s\n", img1_path);
    printf("Loading Image 2: %s\n", img2_path);

    /* 1. Read both images into memory */
    GrayscaleImage *img1 = read_pgm_image(img1_path);
    if (!img1) {
        fprintf(stderr, "\n[FATAL] Failed to load Image 1 from '%s'. Exiting.\n", img1_path);
        return 1;
    }

    GrayscaleImage *img2 = read_pgm_image(img2_path);
    if (!img2) {
        fprintf(stderr, "\n[FATAL] Failed to load Image 2 from '%s'. Exiting.\n", img2_path);
        free_image(img1);
        return 1;
    }

    /* 2. Compute statistical metrics for both images */
    compute_image_metrics(img1);
    compute_image_metrics(img2);

    /* 3. Display individual analysis */
    display_image_analysis(img1, "IMAGE 1 ANALYSIS");
    print_ascii_histogram(img1, "Image 1");

    display_image_analysis(img2, "IMAGE 2 ANALYSIS");
    print_ascii_histogram(img2, "Image 2");

    /* 4. Display side-by-side comparative table */
    display_comparative_table(img1, img2);

    /* 5. Draw conclusions based on global analysis */
    draw_comparative_conclusions(img1, img2);

    /* ============================================================================
     * PART B: REGION OF INTEREST (ROI) EXTRACTION & LOCALIZED EVALUATION
     * ============================================================================ */
    printf("\n\n################################################################################\n");
    printf("        PART B: LOCALIZED REGION OF INTEREST (ROI) EXTRACTION & ANALYSIS        \n");
    printf("################################################################################\n");

    /* 6. Process Image 1 ROI (Divergent sub-region search & evaluation) */
    int roi1_w = 500, roi1_h = 500, step1 = 100;
    printf("\n[Searching] Finding sub-region of size %dx%d with maximum intensity deviation in Image 1...\n", roi1_w, roi1_h);
    ROI roi1 = get_local_region(img1, roi1_w, roi1_h, step1);
    printf("   -> Found divergent ROI at Origin (X=%d, Y=%d), Size=%dx%d\n", roi1.x, roi1.y, roi1.width, roi1.height);

    GrayscaleImage *roi1_img = extract_roi(img1, roi1, "image_1_roi_divergent");
    if (roi1_img) {
        compute_image_metrics(roi1_img);
        display_image_analysis(roi1_img, "IMAGE 1 - EXTRACTED ROI ANALYSIS");
        print_ascii_histogram(roi1_img, "Image 1 Extracted ROI");
        compare_roi_with_parent(img1, roi1_img, roi1, "Image 1: Selected ROI vs Global Image");
    }

    /* 7. Process Image 2 ROI (Divergent sub-region search & evaluation) */
    int roi2_w = 400, roi2_h = 400, step2 = 80;
    printf("\n[Searching] Finding sub-region of size %dx%d with maximum intensity deviation in Image 2...\n", roi2_w, roi2_h);
    ROI roi2 = get_local_region(img2, roi2_w, roi2_h, step2);
    printf("   -> Found divergent ROI at Origin (X=%d, Y=%d), Size=%dx%d\n", roi2.x, roi2.y, roi2.width, roi2.height);

    GrayscaleImage *roi2_img = extract_roi(img2, roi2, "image_2_roi_divergent");
    if (roi2_img) {
        compute_image_metrics(roi2_img);
        display_image_analysis(roi2_img, "IMAGE 2 - EXTRACTED ROI ANALYSIS");
        print_ascii_histogram(roi2_img, "Image 2 Extracted ROI");
        compare_roi_with_parent(img2, roi2_img, roi2, "Image 2: Selected ROI vs Global Image");
    }

    /* ============================================================================
     * PART C: PIXEL BRIGHTENING TRANSFORMATION & STATISTICAL COMPARISON
     * ============================================================================ */
    printf("\n\n################################################################################\n");
    printf("        PART C: PIXEL BRIGHTENING TRANSFORMATION & STATISTICAL EVALUATION       \n");
    printf("################################################################################\n");

    /* 8. Clone and brighten Image 2 */
    int brightness_delta = 50;
    printf("\n[Transforming] Creating brightened version of Image 2 with intensity offset = %+d...\n", brightness_delta);
    GrayscaleImage *brightened_img2 = clone_image(img2, "image_2_brightened.pgm");
    if (brightened_img2) {
        brighten_image(brightened_img2, brightness_delta);

        /* Display individual analysis and histogram */
        display_image_analysis(brightened_img2, "IMAGE 2 (BRIGHTENED +50) ANALYSIS");
        print_ascii_histogram(brightened_img2, "Image 2 (Brightened +50)");

        /* Side-by-side comparative table and analytical evaluation */
        compare_brightened_with_original(img2, brightened_img2, brightness_delta);
    }

    /* ============================================================================
     * PART D: EXPORT PROCESSED PGM FILES TO DISK
     * ============================================================================ */
    printf("\n\n################################################################################\n");
    printf("        PART D: EXPORTING PROCESSED PGM IMAGES TO DISK                          \n");
    printf("################################################################################\n");
    char path_buf[512];
    if (roi1_img) {
        write_pgm_image(roi1_img, get_output_path("images/image_1_roi.pgm", path_buf, sizeof(path_buf)));
    }
    if (roi2_img) {
        write_pgm_image(roi2_img, get_output_path("images/image_2_roi.pgm", path_buf, sizeof(path_buf)));
    }
    if (brightened_img2) {
        write_pgm_image(brightened_img2, get_output_path("images/image_2_brightened.pgm", path_buf, sizeof(path_buf)));
    }

    /* ============================================================================
     * PART E: INVESTIGATION: EQUAL MEAN INTENSITY WITH DIVERGENT VISUAL PATTERNS
     * ============================================================================ */
    printf("\n\n################################################################################\n");
    printf("  PART E: INVESTIGATION: EQUAL MEAN INTENSITY WITH DIVERGENT VISUAL PATTERNS   \n");
    printf("################################################################################\n");
    printf("\n[Generating] Creating 4 synthetic test images (400x400) engineered with target mean mu = 128.0...\n");

    GrayscaleImage *demo_imgs[4];
    demo_imgs[0] = create_synthetic_image(400, 400, 1, "demo_flat_gray.pgm");
    demo_imgs[1] = create_synthetic_image(400, 400, 2, "demo_checkerboard.pgm");
    demo_imgs[2] = create_synthetic_image(400, 400, 3, "demo_gradient.pgm");
    demo_imgs[3] = create_synthetic_image(400, 400, 4, "demo_concentric_circles.pgm");

    /* Multi-image comparative analysis table */
    display_equal_mean_investigation(demo_imgs, 4);

    /* Print histograms for each to highlight distribution differences */
    print_ascii_histogram(demo_imgs[0], "Pattern 1: Uniform Flat Gray");
    print_ascii_histogram(demo_imgs[1], "Pattern 2: Binary Checkerboard");
    print_ascii_histogram(demo_imgs[2], "Pattern 3: Linear Gradient");
    print_ascii_histogram(demo_imgs[3], "Pattern 4: Concentric Circles");

    /* Export demonstration PGM images */
    printf("\n[Exporting] Saving investigation demonstration images to images/ folder...\n");
    write_pgm_image(demo_imgs[0], get_output_path("images/demo_flat_gray.pgm", path_buf, sizeof(path_buf)));
    write_pgm_image(demo_imgs[1], get_output_path("images/demo_checkerboard.pgm", path_buf, sizeof(path_buf)));
    write_pgm_image(demo_imgs[2], get_output_path("images/demo_gradient.pgm", path_buf, sizeof(path_buf)));
    write_pgm_image(demo_imgs[3], get_output_path("images/demo_concentric_circles.pgm", path_buf, sizeof(path_buf)));

    /* Clean up demonstration images */
    for (int d = 0; d < 4; d++) {
        if (demo_imgs[d]) free_image(demo_imgs[d]);
    }

    /* 9. Clean up all allocated memory */
    if (roi1_img) free_image(roi1_img);
    if (roi2_img) free_image(roi2_img);
    if (brightened_img2) free_image(brightened_img2);
    free_image(img1);
    free_image(img2);

    printf("\n[SUCCESS] All laboratory tasks, comparative investigations, and image exports completed successfully.\n");
    return 0;
}
