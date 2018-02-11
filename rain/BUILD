package(
  default_visibility = [
    "//game:__subpackages__"
  ],
)

py_library(
  name = "ai",
  srcs = ["ai.py"],
)

py_test(
  name = "test",
  deps = [":ai"],
  srcs = ["test.py"],
)

py_binary(
  name = "train",
  deps = [":ai"],
  srcs = ["train.py"],
)
