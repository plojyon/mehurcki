import sys

import joblib
import numpy as np


def export_svm(model_path, out_header):
    clf = joblib.load(model_path)

    if clf.kernel not in ("linear", "rbf"):
        raise ValueError("Only linear and RBF kernels supported")

    with open(out_header, "w") as f:
        f.write("#pragma once\n\n")

        # StandardScaler
        if hasattr(clf, "scaler_"):
            mean = clf.scaler_.mean_.astype(np.float16).view(np.uint16)
            scale = clf.scaler_.scale_.astype(np.float16).view(np.uint16)

            f.write("\nstatic const uint16_t scaler_mean[N_FEATURES] = {\n")
            f.write("  " + ", ".join(map(str, mean)) + "\n};\n\n")
            f.write("static const uint16_t scaler_scale[N_FEATURES] = {\n")
            f.write("  " + ", ".join(map(str, scale)) + "\n};\n\n")

            # Sanity check 3
            print("Head of scaler mean should be:")
            print(clf.scaler_.mean_.astype(np.float16)[:4])
            print(clf.scaler_.mean_.astype(np.float16).view(np.uint16)[:4])
            print("Head of scaler scale should be:")
            print(clf.scaler_.scale_.astype(np.float16)[:4])
            print(clf.scaler_.scale_.astype(np.float16).view(np.uint16)[:4])

            f.write("#define USE_SCALER\n")

        # Common parameters
        f.write(f"#define N_FEATURES {clf.support_vectors_.shape[1]}\n")
        f.write(f"#define N_SUPPORT {clf.support_vectors_.shape[0]}\n\n")

        # Support vectors
        f.write("static const uint16_t support_vectors[N_SUPPORT][N_FEATURES] = {\n")
        for sv in clf.support_vectors_.astype(np.float16).view(np.uint16):
            f.write("  {" + ", ".join(map(str, sv)) + "},\n")
        f.write("};\n\n")
        # Sanity check 1
        print("Head of support_vectors[0] should be:")
        print(clf.support_vectors_.astype(np.float16)[0][:4])
        print(clf.support_vectors_.astype(np.float16).view(np.uint16)[0][:4])

        # Dual coefficients
        coef = clf.dual_coef_[0].astype(np.float16).view(np.uint16)
        f.write("static const uint16_t dual_coef[N_SUPPORT] = {\n")
        f.write("  " + ", ".join(map(str, coef)) + "\n};\n\n")
        # Sanity check 2
        print("Head of dual coefficients should be:")
        print(clf.dual_coef_[0].astype(np.float16)[:400])
        print(clf.dual_coef_[0].astype(np.float16).view(np.uint16)[:400])

        # Intercept
        f.write(f"static const float intercept = {clf.intercept_[0]};\n\n")
        print(f"Intercept (sussy number) is {clf.intercept_[0]}")

        if clf.kernel == "rbf":
            f.write(f"#define GAMMA {clf._gamma}\n")
            print(f"Gamma is {clf._gamma}")
            f.write("#define KERNEL_RBF\n")
        else:
            f.write("#define KERNEL_LINEAR\n")

        f.write("\n// End of SVM parameters\n")

if __name__ == "__main__":
    print(f"Exporting {sys.argv[1]}")
    export_svm(sys.argv[1], "esp/detector_params.h")
