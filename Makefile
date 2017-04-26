CFLAGS = -msse2 --std gnu99 -O0 -Wall -Wextra -mavx
EXEC = naive sse sse_prefetch avx avx_prefetch

GIT_HOOKS := .git/hooks/applied

SRCS_common = main.c

all: $(GIT_HOOKS) $(EXEC)

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo


naive: $(SRCS_common)
	$(CC) $(CFLAGS) -D$@ -DName=\"$@\" -o $@ $(SRCS_common)
sse: $(SRCS_common)
	$(CC) $(CFLAGS) -D$@ -DName=\"$@\" -o $@ $(SRCS_common)
sse_prefetch: $(SRCS_common)
	$(CC) $(CFLAGS) -D$@ -DName=\"$@\" -o $@ $(SRCS_common)
avx: $(SRCS_common)
	$(CC) $(CFLAGS) -D$@ -DName=\"$@\" -o $@ $(SRCS_common)
avx_prefetch: $(SRCS_common)
	$(CC) $(CFLAGS) -D$@ -DName=\"$@\" -o $@ $(SRCS_common)


clean:
	$(RM) $(EXEC)
