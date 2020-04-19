git clone --recursive --single-branch --branch feature/catsvsdogs https://github.com/iotwithit/openmv
cp build/libtf_catsvsdogs_classify_model_data.a openmv/src/libtf/cortex-m7/
cd openmv/src/micropython/mpy-cross
make -j$(nproc)
cd ../..
make -j$(nproc)
