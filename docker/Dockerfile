FROM python:3.10.10-bullseye

# build-time variables to be set by the Makefile
ARG group
ARG gid
ARG user
ARG uid
RUN groupadd --gid $gid $group && \
    useradd --create-home --shell /bin/bash --gid $gid --uid $uid $user

USER $user

CMD ["/bin/bash"]
