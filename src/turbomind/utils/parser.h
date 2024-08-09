#include <string>
#include <vector>

namespace turbomind {

std::vector<std::pair<std::string, std::string>> ParseArgsList(const std::string& str);

std::vector<std::string> ParseListOrTuple(const std::string& str);

inline void Parse(int& value, const std::string& str)
{
    value = std::stoi(str);
}

inline void Parse(float& value, const std::string& str)
{
    value = std::stof(str);
}

template<class T>
void Parse(std::vector<T>& xs, const std::string& str)
{
    const auto ss = ParseListOrTuple(str);
    for (const auto& s : ss) {
        xs.emplace_back();
        Parse(xs.back(), s);
    }
}

}  // namespace turbomind
