import requests
from math import radians, cos, sin, asin, sqrt


def get_position_by_name(address):
    """Get longitude and lattitude of a city by baidu api
    """
    url = f'http://api.map.baidu.com/geocoder?output=json&key=f247cdb592eb43ebac6ccd27f796e2d2&address={address}&' \
          f'city={address}'
    response = requests.get(url)
    # print(address)
    # print(response)
    # print(response.json())
    answer = response.json()
    # print(address + "的经纬度：", answer['geocodes'][0]['location'])
    if answer['status'] == 'INVALID_PARAMETERS':
        return 0, 0
    else:
        lon = float(answer['result']['location']['lng'])
        lat = float(answer['result']['location']['lat'])
        return lon, lat


def compute_distance(address1, address2):
    """Compute the distance between two given city

    @:return distance, the unit is kilometer
    """
    lon1, lat1 = get_position_by_name(address1)
    lon2, lat2 = get_position_by_name(address2)

    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return int(c * r)


if __name__ == "__main__":
    print(compute_distance("北京市", "莫斯科"))
    # print(geocode("悉尼"))