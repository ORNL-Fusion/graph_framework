//------------------------------------------------------------------------------
///  @file absoprtion.hpp
///  @brief Base class for a dispersion relation.
///
///  Defines functions for computing power absorbtion.
//------------------------------------------------------------------------------

#ifndef absorption_h
#define absorption_h

namespace absorption {
//******************************************************************************
//  Root finder.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class interface for the root finder.
///
///  @tparam DISPERSION_FUNCTION Class of dispersion function to use.
//------------------------------------------------------------------------------
    template<class DISPERSION_FUNCTION>
    class root_finder {
    };
}

#endif /* absorption_h */
