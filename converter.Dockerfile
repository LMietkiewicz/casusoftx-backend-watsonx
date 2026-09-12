# =============================================================================
# LibreOffice conversion sidecar, s390x.
#
# Exists because the app's base image is RHEL 10, which no longer packages
# LibreOffice, and Gotenberg publishes no s390x image. Ubuntu builds LibreOffice
# for s390x as an official architecture (noble-updates).
#
# unoserver keeps ONE LibreOffice process resident and exposes it over XML-RPC
# on port 2003. That avoids both the ~1.2s per-conversion startup and the
# profile-lock contention that makes concurrent 'soffice' invocations fail
# silently (measured: 2 of 4 parallel jobs failed, no useful error).
#
# Runs as the second container in the backend pod (see openshift-manifests.yaml)
# or as the `converter` service in docker-compose.test.yml.
#
# Build:  docker build -f converter.Dockerfile -t csx-converter .
# =============================================================================

FROM ubuntu:24.04

# HOME must be writable: the OpenShift arbitrary UID has no /etc/passwd entry,
# and LibreOffice writes its user profile under HOME. /tmp is an emptyDir in
# the pod. DEBIAN_FRONTEND stops apt from prompting during the build.
ENV HOME=/tmp \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# libreoffice-writer covers .doc/.docx/.txt/.xml (all Writer formats).
# python3-uno is the LibreOffice Python bridge unoserver drives; it targets the
# system python3, which is why unoserver is installed into that interpreter and
# not a venv.
# fonts-*: with --no-install-recommends LibreOffice can end up with no usable
# font and renders missing glyphs as boxes. DejaVu and Liberation both carry
# full Polish diacritics and are the fonts .docx files most commonly reference.
#
# unoserver is pinned to the SAME version as the client in requirements.txt:
# client and server share a versioned XML-RPC API.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer python3-uno python3-pip \
        fonts-dejavu fonts-liberation \
    && pip3 install --no-cache-dir --break-system-packages unoserver==3.7 \
    && rm -rf /var/lib/apt/lists/*

# Arbitrary-UID fixup: the container user is a member of GID 0 whatever UID
# it gets; make the profile location writable to that group.
RUN chgrp -R 0 /tmp && chmod -R g=u /tmp

# Numeric, never a username (OpenShift SCC validation).
USER 1001
EXPOSE 2003

# UNOSERVER_INTERFACE selects who can reach the converter:
#   127.0.0.1 (default)  pod sidecar: shares the backend's network namespace,
#                        so loopback is reachable and nothing else is.
#   0.0.0.0              docker-compose: separate container, reached by service
#                        name, so it must listen on all interfaces.
# Shell form so the variable expands; `exec` keeps unoserver as PID 1 so it
# receives SIGTERM directly on pod shutdown.
CMD exec unoserver --port 2003 --uno-port 2002 --interface "${UNOSERVER_INTERFACE:-127.0.0.1}"
