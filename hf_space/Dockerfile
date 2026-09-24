# Hugging Face Docker Space: the full FlyWire fly brain (Shiu et al. 2024) as a WebSocket server
# for the zvuv control site. The code is pulled from GitHub at build time, so updating the
# Space = "Factory rebuild" in the Space settings. No other files need to be uploaded.
# The full python image already has git and g++ (Brian2 compiles the model to C++,
# about 6x faster than pure numpy), so no apt-get step is needed
FROM python:3.12

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

ARG REPO=itaydaniel1000-beep/zvuv
ARG BRANCH=main
# "Factory rebuild" in the Space settings builds without cache, so it always gets the latest code
RUN git clone --depth 1 --branch ${BRANCH} https://github.com/${REPO}.git zvuv

RUN pip install --no-cache-dir --user -r zvuv/brain_server/requirements.txt

# Download the FlyWire data (~140 MB) and compile the model once, at build time,
# so the Space starts in seconds instead of minutes
RUN python zvuv/brain_server/server.py --selftest --target cython --step 10

EXPOSE 7860
CMD ["python", "zvuv/brain_server/server.py", "--host", "0.0.0.0", "--port", "7860", "--target", "cython"]
