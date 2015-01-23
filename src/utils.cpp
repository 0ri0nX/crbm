#include "utils.h"
#include "matrixCpu.h"

using namespace YAMATH;

void loadMatrix(YAMATH::MatrixCpu &inM, const std::string& filename, bool inTransposed)
{
#if LOAD_VERBOSITY > 1
    std::cout << "loading [" << filename << "] ... " << std::flush;
    Timer t;
#endif

    MatrixLoaderFile loader(filename);
    loader.LoadComplete(inM, inTransposed);

#if LOAD_VERBOSITY > 1
    std::cout << inM.getX() << " x " << inM.getY() << "  ";
    t.tac();
#endif
}

void saveMatrix(const YAMATH::MatrixCpu &inM, const std::string &filename, int inVersion)
{
    std::cout << "saving [" << filename << "] ... " << std::flush;
    Timer t;
    MatrixSaverFile saver(filename, inVersion);
    saver.SaveComplete(inM);
    t.tac();
}

//load from stream (cpu matrix)
template<>
void lv<>(std::istream &in, const std::string &inName, YAMATH::MatrixCpu &outValue)
{
#if LOAD_VERBOSITY > 1
    std::cout << "Loading [" << inName << "] ..." << std::flush;
#endif

    std::string name;
    in >> name;
    in.ignore(1);
    assert(name == inName);

    outValue.Load(in);

#if LOAD_VERBOSITY > 1
    std::cout << " done" << std::endl;
#endif
}

//save to stream (cpu matrix)
template<>
void sv<>(std::ostream &out, const std::string &inName, const YAMATH::MatrixCpu &inValue)
{
    out << inName << " ";
    inValue.Save(out);
    out << std::endl;
}

