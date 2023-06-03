
# 没有信息的用户id
# 57 => 83 / 420

no_information_users = [2, 3, 9, 23, 35, 39, 40, 55, 56, 64, 65, 68, 79, 80, 81, 88, 89, 90, 91, 92, 93, 94, 100, 109,
 127, 133, 142, 148, 155, 157, 160, 161, 174, 179, 189, 194, 304, 308, 315, 324, 325, 327, 338,
 345, 362, 375, 382, 383, 393, 394, 396, 397, 403, 410, 416, 417, 419]

# 139 / 1000
no_information_services = [
	10, 16, 17, 18, 19, 20, 53, 54, 55, 60, 67, 68, 70, 76, 90, 91, 92, 106, 107, 113, 135, 148, 149, 150,
	167, 168, 169, 170, 173, 174, 196, 200, 203, 210, 211, 212, 533, 534, 535, 542, 543, 547, 548, 549, 550,
	556, 557, 559, 560, 561, 562, 563, 564, 565, 590, 593, 594, 599, 600, 601, 602, 603, 604, 605, 606, 620,
	621, 622, 623, 624, 660, 661, 670, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 722, 723, 724, 725,
	730, 731, 732, 733, 734, 737, 746, 747, 748, 761, 768, 769, 782, 783, 792, 800, 804, 808, 809, 812, 815,
	818, 822, 828, 835, 838, 839, 840, 841, 854, 856, 860, 862, 903, 912, 919, 921, 936, 937, 945, 947, 948,
	951, 952, 957, 958, 968, 969, 972, 974, 987, 990
]

user_num = 420
item_num = 1000

countrys = [
	'Bosnia and Herzegovina', 'Netherlands', 'Greece ', 'Czech Republic ', 'Belarus', 'United Kingdom', 'New Zealand', 'India',
	'Chile', 'Portugal', 'Hong Kong ', 'Israel', 'Israel ', 'Virgin Islands, British', 'Greece', 'Switzerland ', 'Spain ',
	'Belgium', 'Romania', 'Bangladesh', 'United States', 'Costa Rica', 'Korea', 'Cyprus', 'Brazil', 'Thailand', 'Bulgaria',
	'Spain', 'Canada ', 'Finland', 'Denmark', 'Australia', 'Russian Federation', 'Croatia', 'None', 'Hong Kong', 'Italy ',
	'Hungary', 'Sweden', 'Slovenia', 'China ', 'Argentina', 'Austria', 'Uruguay', 'Switzerland', 'Norway', 'Singapore',
	'United Kingdom ', 'Germany', 'Poland ', 'Canada', 'Poland', 'France', 'Norway ', 'Iceland', 'Turkey', 'Germany ',
	'Japan', 'Czech Republic', 'Ireland', 'Italy', 'United States ', 'Portugal ', 'Puerto Rico', 'Slovenia ', 'China'
]

citys = [
	'Rostock', 'West University Place', 'Seoul', 'ChangSha', 'San Ramon', 'Wellington', 'Tarragona', 'Singapore', 'Mianyang',
	'Prague', 'Auckland', 'Burnaby', 'Ballerup', 'Lowell', 'Ironsides', 'Copenhagen', 'Banja Luka', 'Cambridge', 'Shenyang',
	'Montville Center', 'Brno', 'Zhengzhou', 'Linz', 'Neochoropoulon', 'Zhoushan', 'Gastonville', 'Dongguan', 'Windcrest',
	'Port Tobacco', 'Langfang', 'Worcester', 'Liverpool', 'Catania', 'Wuerzburg', 'Newark', 'Santa Barbara', 'Lisbon', 'Bangkok',
	'Druid Hills', 'Osaka', 'La Jolla', 'Findon', 'Zhumadian', 'Iquique', 'Helsinki', 'Ljubljana', 'Brussels', 'Gudang', 'Florida',
	'Guangzhou', 'Zelenograd', 'Montreal', 'Santiago', 'Oststadt', 'Minsk', 'Lincoln', 'Scottsdale', 'Queen Creek', 'Trento',
	'Williamstown', 'Waterloo', 'Liaoyang', 'Mumbai', 'Moosburg', 'Cordoba', 'Jasper', 'Belo Horizonte', 'East Cleveland',
	'Sevlievo', 'Plzen', 'Barcelona', 'Changsha', 'Joao Pessoa', 'Clayton North', 'Yangzhou', 'New York City', 'Birmingham',
	'Biot', 'Sao Bernardo do Campo', 'London', 'Rennes', 'Coton', 'Pamplona', 'Charlottesville', 'Indooroopilly', 'Winnipeg',
	'Quebec', 'Trenton', 'Shanghai', 'Surry Hills', 'Castelar', 'Baile Atha Luain', 'Lawrence', 'Taiyuan', 'Chicago', 'Ann Arbor',
	'Aveiro', 'Evry', 'Bloomington', 'Traralgon', 'Hangzhou', 'Brisbane', 'Antwerpen', 'Philadelphia', 'Lexington-Fayette', 'Suwon',
	'Nack', 'Oostende', 'Provo', 'Seattle', 'Curitiba', 'Petah Tikva', 'Rome', 'Opole', 'Houghton', 'Clemson', 'Toronto', 'Linqiong',
	'Dresden', 'Skanderborg', 'Nicosia', 'Sao Paulo', 'Pomona', 'Gent', 'None', 'Suzhou', 'Macquarie', 'Reno', 'Austin', 'Victoria',
	'New Haven', 'Frankfurt ', 'Zurich', 'Athens', 'Brooklyn', 'Xiamen', 'Zwettl', 'Alegre', 'Edgewater', 'Dusseldorf', 'Dublin',
	'Zaozhuang', 'Linkoping', 'Wuxi', 'Diamond Bar', 'Baltimore', 'Gainesville', 'Melbourne', 'San Mateo', 'Tokyo', 'Warsaw',
	'Gottingen', 'de Buenos Aires', '-', 'Montevideo', 'Mendoza', 'Salzburg', 'Sofia', 'Issy-les-Moulineaux', 'Darmstadt', 'Lodz',
	'Brighton', 'Graz', 'North Kensington', 'Koge', 'Toowoomba', 'Reykjavik', 'Blacksburg', 'College Park', 'San Luis Obispo',
	'Sydney', 'Madrid', 'Koeln', 'Shenzhen', 'Dunedin', 'Troyes', 'Road Town', 'Amsterdam', 'Neuchatel', 'Hamburg', 'Udine',
	'Porto Belo', 'Miami', 'Targu-Mures', 'Odense', 'Ede', 'Curico', 'Valparaiso', 'Mechelen', 'Fairfax', 'Rio de Janeiro',
	'Poznan', 'College Station', 'Ryde', 'Parma', 'Providence', 'Hiroshima', 'Chongqing', 'Reston', 'Campion', 'Urbana', 'Munich',
	'Nanning', 'Randers', 'College Statio', 'San Jose', 'Anshan', 'Wilcox Corners', 'Portland', 'Phoenix', 'Berkeley', 'Jinan',
	'Basel', 'Plano', 'Namur', 'Ottawa', 'Tromso', 'Louvain-la-Neuve', 'Seraing', 'Zagreb', 'Samobor', 'Medicine Hat', 'Salt Lake City',
	'Bathurst', 'Chapel Hill', 'Oostrozebeke', 'Lulea', 'Menlo Park', 'Los Angeles', 'Saint Andrews', 'Ashburn', 'Qingdao', 'Kunming',
	'Tel Aviv', 'Dalianwan', 'Espoo', 'Stillwater', 'East Ithaca', 'Taiden', 'Mapleton', 'Vienna', 'Highland Park', 'Saskatoon',
	'San Juan', 'Tampa', 'Pennsylvania', 'Braga', 'Murcia', 'Frederiksberg', 'Dallas', 'Naples', 'Zlin', 'Atlanta', 'Wroclaw',
	'Zhongshan', 'Rion', 'Riverside', 'Kongens Lyngby', 'San Francisco', 'Tianjin', 'Foshan', 'Fuzhou', 'Hoboken', 'Ayr',
	'Nashville', 'Foz do Iguacu', 'Almada', 'Brasilia', 'Buenos Aires', 'Stuttgart', 'Orlando', 'Dhaka', 'Macquarie Park',
	'Troms', 'Leuven', 'Edmonton', 'Izmir', 'Budapest', 'Kerkyra', 'Varde', 'Burleigh Heads', 'Sankt Magnus', 'ByWard Market',
	'Wuhan', 'Colonie', 'Zhaoqing', 'Beijing', 'West Lafayette', 'Joliette', 'Hamilton', 'Amherst', 'Vancouver', 'Hong Kong',
	'Faro', 'Vantaa', 'Rijeka', 'Stockholm', 'Passau', 'Ostrava', 'Walnut', 'Frankfurt am Main', 'Grandview Heights', 'Berlin',
	'Town and Country', 'Ningbo', 'Boston', 'Istanbul', 'Goiania', 'Kansas City', 'Erlangen', 'Oslo', 'Montrea'
]

useless_user = []
for user_id in range(0, 420):
	useless_user.append(0)
for user_id in no_information_users:
	useless_user[user_id] = 1

useless_service = []
for service_id in range(0, 1000):
	useless_service.append(0)
for service_id in no_information_services:
	useless_service[service_id] = 1

def binary_search(arr, num):
	start = 0
	end = len(arr)-1
	while (start <= end):
		mid = (start + end) / 2
		mid = int(mid)
		midval = arr[mid]
		if midval == num:
			return True
		elif midval > num:
			end = mid - 1
		else:
			start = mid + 1
	return False


def binary_search_user(num):
	return useless_user[num] == 1
	# return binary_search(no_information_users, num)

def binary_search_service(num):
	return useless_service[num] == 1
	# return  binary_search(no_information_services, num)

def count_country_and_city():
	country_set = set()
	city_set = set()
	for line in open("./userlist.txt", 'r'):
		arr = line.replace("\n", '').split("\t")
		country_set.add(arr[2])
		city_set.add(arr[4])

	for line in open("./wslist.txt", 'r'):
		arr = line.replace("\n", '').split("\t")
		country_set.add(arr[2])
		city_set.add(arr[4])