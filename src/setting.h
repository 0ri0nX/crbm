#ifndef SETTING_H
#define SETTING_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cassert>
#include <iomanip>

struct SettingBase
{
    SettingBase()
    {
    }

    enum Type
    {
        Float,
        Int,
    };

    template<typename T>
    T loadOption(std::ifstream &f, const char* inOption, T inDefault)
    {
        f.clear();
        f.seekg(0);

        T res = inDefault;
        bool loaded = false;

        while(f.good())
        {
            const int maxSize = 1024;
            char buffer[maxSize];
            char buffer2[maxSize];

            f.getline(buffer, maxSize);
            if(buffer[0] != '#' && strlen(buffer) > 0)
            {
                std::stringstream s(buffer);
                s.getline(buffer2, maxSize, ' ');
                assert(strlen(buffer2) < maxSize-1);

                //std::cout << inOption << ": " << buffer << ": " << buffer2 << std::endl;

                if(strcmp(buffer2, inOption) == 0)
                {
                    T value;
                    s >> value;
                    if(!s.fail())
                    {
                        res = value; 
                        loaded = true;
                        break;
                    }
                }
            }
        }

        std::cout << std::setw(20) << std::right << inOption << " = " << std::setw(10) << std::left << res;
        std::cout << (loaded ? " (loaded)" : " (default)") << std::endl;
        return res;
    }


    void loadFromFile(const char *inConfigName)
    {
        std::ifstream f(inConfigName);

        std::cout << "Loading configuration from file [" << inConfigName << "]" << std::endl;

        loadFromStream(f);

        f.close();
    }

    virtual void loadFromStream(std::ifstream &f) = 0;

};

#endif // SETTING_H

