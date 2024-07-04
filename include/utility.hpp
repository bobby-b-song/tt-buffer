#include <filesystem>

inline std::string get_absolute_path(std::string path) {
  return std::filesystem::absolute(path);
}
