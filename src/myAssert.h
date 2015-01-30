#ifndef MYASSERT_H
#define MYASSERT_H

#include <exception>
#include <sstream>

#define ASSERT(x) if (!(x)) { std::stringstream msg; msg << "Error: " #x " in file " __FILE__ ":" << __LINE__; throw std::runtime_error(msg.str()); }

#endif // MYASSERT_H

