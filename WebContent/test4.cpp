#include <iostream>
#include <vector>
#include <map>

int main() {
    // 定义一个序列，其中每个元素是一个 map 结构的元素
    std::vector<std::map<int, std::string>> vectorOfMaps;

    // 向 vector 中添加 map 元素
    std::map<int, std::string> map1 = {{1, "apple"}, {2, "banana"}, {3, "orange"}};
    vectorOfMaps.push_back(map1);

    std::map<int, std::string> map2 = {{4, "grape"}, {5, "pineapple"}};
    vectorOfMaps.push_back(map2);

    // 遍历 vector 中的每个 map
    for (const auto& map : vectorOfMaps) {
        // 遍历 map 中的键值对
        for (const auto& pair : map) {
            std::cout << pair.first << ": " << pair.second << std::endl;
        }
    }

    return 0;
}