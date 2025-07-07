#include "csv_drawer.h"
#include <iostream>
#include "../rubbish_can.h" // 假设这是您的垃圾回收工具头文件

int main() {
    draw::csv_drawer drawer("test_output.csv");
    
    draw::statistic test_data1 = {
        {"temperature", 25.3f},
        {"pressure", 1013.2f},
        {"humidity", 60.5f}
    };
    
    print_unordered_map(test_data1);
    
    drawer.draw(test_data1);
    
    draw::statistic test_data2 = {
        {"temperature", 26.1f},
        {"pressure", 1012.8f},
        {"humidity", 58.9f}
    };
    
    print_unordered_map(test_data2);
    drawer.draw(test_data2);
    
    std::cout << "CSV drawer test completed. Check test_output.csv" << std::endl;
    return 0;
}
