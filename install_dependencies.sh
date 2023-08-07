sudo apt install ffmpeg

pip install --force-reinstall torch torchvision torchaudio

pip install pydub
pip install datasets

pip install transformers
pip install librosa
pip install soundfile
pip install evaluate
pip install jiwer
pip install tensorboardX
pip install accelerate

pip install wandb

# cuda-cudart-11.7.99-0
#   cuda-cupti-11.7.101-0
#   cuda-libraries-11.7.1-0
#   cuda-nvrtc-11.7.99-0
#   cuda-nvtx-11.7.91-0
#   cuda-runtime-11.7.1-0
#   cudatoolkit-11.7.0-hd8887f6_11

# whisperx examples/F-285.wav --model large-v2 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 4 --compute_type int8 --output_dir diari --language es --diarize --min_speakers 2 --max_speakers 3 --hf_token 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

pip install sentence_transformers
pip install chromadb==0.3.29
