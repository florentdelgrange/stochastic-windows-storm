//
// Created by Florent Delgrange on 30/10/2018.
//

#include <iostream>
#include "FooBar.h"

int sw::FooBarObject::foo() {

    std::cout << "SALUTATIONS" << std::endl;

    return 0;
}

sw::FooBarObject::FooBarObject() {

    std::cout << "Foo bar object initialized" << std::endl;

}
