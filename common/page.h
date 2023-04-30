#ifndef PD_COMMON_PAGE_H_
#define PD_COMMON_PAGE_H_
#include <string>
#include <vector>
namespace pd_common
{

    struct PdPage
    {
        std::string title;
        std::string url;
        std::string content;

        // filled by dedup
        std::vector<int> min_hashes;
    };

} // namespace pd_common

#endif // PD_COMMON_PAGE_H_
