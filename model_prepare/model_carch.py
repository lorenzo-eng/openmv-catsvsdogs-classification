import subprocess
import config

cc = """
#include "libtf_catsvsdogs_classify_model_data.h"

// We need to keep the data array aligned on some architectures.
#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif


const unsigned char g_catsvsdogs_classify_model_data[] DATA_ALIGN_ATTRIBUTE = {{
{}
}};
const int g_catsvsdogs_classify_model_data_len = {};
"""

with open(config.TFLITE_MODEL_PATH, "rb") as rr:
    bin_model = rr.read()

data_model = "    "
for bi, bb in enumerate(bin_model):
    data_model += "0x%02x," % bb
    data_model += "\n    " if bi % 12 == 11 else " "

cc = cc.format(data_model, len(bin_model))
with open("csrc/libtf_catsvsdogs_classify_model_data.cc", "w+") as ww:
    ww.write(cc)

compile_cmd = "arm-none-eabi-gcc -D __FPU_PRESENT=1 -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -DNDEBUG -DTF_LITE_DISABLE_X86_NEON -DTF_LITE_MCU_DEBUG_LOG -DTF_LITE_STATIC_MEMORY -MMD -O3 -Wall -Wextra -Wvla -Wno-format -Wno-missing-field-initializers -Wno-parentheses -Wno-sign-compare -Wno-strict-aliasing -Wno-type-limits -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-variable -Wno-write-strings -fdata-sections -ffunction-sections -fmessage-length=0 -fomit-frame-pointer -funsigned-char -fno-builtin -fno-delete-null-pointer-checks -fno-exceptions -fno-unwind-tables -mfloat-abi=hard -mlittle-endian -mthumb -mno-unaligned-access -nostdlib -std=c++11 -std=gnu++11 -fno-rtti -fpermissive -DARM_CMSIS_NN_M7 -DARM_MATH_CM7 -mcpu=cortex-m7 -mfpu=fpv5-sp-d16 -mtune=cortex-m7 -o csrc/libtf_catsvsdogs_classify_model_data.o -c csrc/libtf_catsvsdogs_classify_model_data.cc"
ar_cmd = "arm-none-eabi-ar rcs %s/libtf_catsvsdogs_classify_model_data.a csrc/libtf_catsvsdogs_classify_model_data.o" % config.OUTPUT_DIR
clean_cmd = "rm csrc/libtf_catsvsdogs_classify_model_data.o csrc/libtf_catsvsdogs_classify_model_data.d"

print("compile.")
subprocess.run(compile_cmd, shell=True)
print("archive.")
subprocess.run(ar_cmd, shell=True)
print("clean.")
subprocess.run(clean_cmd, shell=True)