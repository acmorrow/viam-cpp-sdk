#include <memory>
#define BOOST_TEST_MODULE test module test_types

#include <unordered_map>

#include <boost/test/included/unit_test.hpp>

#include <viam/sdk/common/proto_type.hpp>
#include <viam/sdk/config/resource.hpp>
#include <viam/sdk/tests/test_utils.hpp>

namespace viam {
namespace sdktests {

using namespace viam::sdk;

BOOST_AUTO_TEST_SUITE(test_prototype)

BOOST_AUTO_TEST_CASE(test_prototype_equality) {
    AttributeMap expected_map = fake_map();

    ProtoType type1 = ProtoType(expected_map);

    AttributeMap expected_map2 = fake_map();

    ProtoType type2 = ProtoType(expected_map2);

    BOOST_CHECK(type1 == type2);

    auto proto_ptr = std::make_shared<ProtoType>(std::move(std::string("not hello")));
    AttributeMap unequal_map =
        std::make_shared<std::unordered_map<std::string, std::shared_ptr<ProtoType>>>();
    unequal_map->insert({{std::string("test"), proto_ptr}});

    ProtoType type3 = ProtoType(unequal_map);

    BOOST_CHECK(!(type1 == type3));
}

BOOST_AUTO_TEST_CASE(test_prototype_list_conversion) {
    std::string s("string");
    double d(3);
    bool b(false);
    std::vector<std::shared_ptr<ProtoType>> proto_vec{std::make_shared<ProtoType>(d),
                                                      std::make_shared<ProtoType>(s),
                                                      std::make_shared<ProtoType>(b)};

    ProtoType proto(proto_vec);

    google::protobuf::Value double_value;
    double_value.set_number_value(d);
    google::protobuf::Value string_value;
    string_value.set_string_value(s);
    google::protobuf::Value bool_value;
    bool_value.set_bool_value(b);
    google::protobuf::ListValue lv;

    *lv.add_values() = double_value;
    *lv.add_values() = string_value;
    *lv.add_values() = bool_value;

    google::protobuf::Value v;
    *v.mutable_list_value() = lv;

    ProtoType proto_from_value(v);

    BOOST_CHECK_EQUAL(proto.proto_value().list_value().values_size(), 3);
    BOOST_CHECK(proto == proto_from_value);

    auto round_trip1 = proto.proto_value();
    auto round_trip2 = ProtoType(round_trip1);

    BOOST_CHECK(round_trip2 == proto);
}

BOOST_AUTO_TEST_CASE(test_prototype_map_conversion) {
    std::string s("string");
    double d(3);
    bool b(false);

    auto m = std::make_shared<std::unordered_map<std::string, std::shared_ptr<ProtoType>>>();

    m->insert({{std::string("double"), std::make_shared<ProtoType>(d)}});
    m->insert({{std::string("bool"), std::make_shared<ProtoType>(b)}});
    m->insert({{std::string("string"), std::make_shared<ProtoType>(s)}});

    google::protobuf::Value double_value;
    double_value.set_number_value(d);
    google::protobuf::Value string_value;
    string_value.set_string_value(s);
    google::protobuf::Value bool_value;
    bool_value.set_bool_value(b);
    google::protobuf::Map<std::string, google::protobuf::Value> proto_map;
    proto_map.insert({{std::string("string"), string_value}});
    proto_map.insert({{std::string("double"), double_value}});
    proto_map.insert({{std::string("bool"), bool_value}});

    google::protobuf::Struct proto_struct;
    *proto_struct.mutable_fields() = proto_map;
    AttributeMap from_proto = struct_to_map(proto_struct);
    BOOST_CHECK_EQUAL(from_proto->size(), m->size());

    ProtoType proto(from_proto);
    ProtoType map(m);
    BOOST_CHECK(map == proto);

    auto round_trip1 = map_to_struct(m);
    auto round_trip2 = struct_to_map(round_trip1);

    ProtoType round_trip_proto(round_trip2);
    BOOST_CHECK(round_trip_proto == map);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace sdktests
}  // namespace viam

namespace magic {

struct object_tag;
struct array_tag;

template <typename T>
struct recursive_variant_wrapper_traits;

template <typename T>
class recursive_variant_wrapper {
   public:
    using type = typename recursive_variant_wrapper_traits<T>::type;

    recursive_variant_wrapper() = delete;

    recursive_variant_wrapper(const type& t) : impl_(std::make_unique<type>(t)) {}

    recursive_variant_wrapper(type&& t) : impl_(std::make_unique<type>(std::move(t))) {}

    template <typename U, typename = std::enable_if_t<std::is_convertible<U, T>::value>>
    recursive_variant_wrapper(U&& u) : impl_(std::make_unique<type>(std::forward<U>(u))) {}

    recursive_variant_wrapper(const recursive_variant_wrapper& other)
        : impl_(std::make_unique<type>(*other.impl_)) {}

    template <typename U>
    recursive_variant_wrapper(const recursive_variant_wrapper<U>& other)
        : impl_(std::make_unique<type>(*other.impl_)) {}

    recursive_variant_wrapper(recursive_variant_wrapper&& other) noexcept(false)
        : impl_(std::make_unique<type>(std::move(*other.impl_))) {}

    template <typename U>
    recursive_variant_wrapper(recursive_variant_wrapper<U>&& other)
        : impl_(std::make_unique<type>(std::move(*other.impl_))) {}

    recursive_variant_wrapper& operator=(const type& t) {
        *impl_ = t;
        return *this;
    }

    recursive_variant_wrapper& operator=(type&& t) {
        *impl_ = std::move(t);
        return *this;
    }

    template <typename U, typename = std::enable_if_t<std::is_convertible<U, T>::value>>
    recursive_variant_wrapper& operator=(U&& u) {
        *impl_ = std::forward<U>(u);
        return *this;
    }

    recursive_variant_wrapper& operator=(const recursive_variant_wrapper& other) {
        if (this != &other) {
            *impl_ = *other._impl;
        }
        return *this;
    }

    template <typename U>
    recursive_variant_wrapper& operator=(const recursive_variant_wrapper<U>& other) {
        *impl_ = *other._impl;
        return *this;
    }

    recursive_variant_wrapper& operator=(recursive_variant_wrapper&& other) noexcept {
        *impl_ = std::move(*other._impl);
    }

    template <typename U>
    recursive_variant_wrapper& operator=(recursive_variant_wrapper<U>&& other) {
        *impl_ = std::move(*other._impl);
        return *this;
    }

    operator type&() & {
        return *impl_;
    };

    operator const type&() const& {
        return *impl_;
    }

    operator type&&() && {
      return std::move(*impl_);
    }

    void swap(recursive_variant_wrapper& other) noexcept {
      impl_.swap(other._impl);
    }

    friend void swap(recursive_variant_wrapper& l, recursive_variant_wrapper& r) noexcept {
        l.swap(r);
    }

   private:
    std::unique_ptr<type> impl_;
};

using value = std::variant<int,
                           bool,
                           std::string,
                           recursive_variant_wrapper<array_tag>,
                           recursive_variant_wrapper<object_tag>>;

template <>
struct recursive_variant_wrapper_traits<object_tag> {
    using type = std::unordered_map<std::string, value>;
};

template <>
struct recursive_variant_wrapper_traits<array_tag> {
    using type = std::vector<value>;
};

using object = recursive_variant_wrapper_traits<object_tag>::type;
using array = recursive_variant_wrapper_traits<array_tag>::type;


void testme() {
    object o;
    o.insert({"key", 42});

    object o2;
    o.insert({"key2", std::move(o2)});

    o.insert({"stringy", "McStringString"});

    o.insert({"false", false});
    o.insert({"true", true});

    std::string x;

    value v{std::move(o)};
    auto xx = std::get<object>(v);
}

}  // namespace magic
