"""
Print registration renewal invoices for all vehicles.

Supports all subclasses of Vehicle.
"""

from my_vehicles import Car, Truck, Motorcycle


def main():
    vehicles = get_vehicles()
    vehicles.sort(key=by_first_name)
    vehicles.sort(key=by_last_name)
    vehicles.sort(key=by_city)
    print_invoices(vehicles)


def get_vehicles():
    data_directory_name = 'data'
    infile_name = input('Please enter the input filename: ')
    infile_path_and_name = f'{data_directory_name}/{infile_name}'
    infile = open(infile_path_and_name, 'r', encoding='utf-8')
    my_vehicles = []
    for line in infile:
        if line.startswith('Car'):
            vehicle = construct_car_instance(line)
        elif line.startswith('Truck'):
            vehicle = construct_truck_instance(line)
        elif line.startswith('Motorcycle'):
            vehicle = construct_motorcycle_instance(line)
        else:
            raise ValueError(f'0 invoices have been printed')
        my_vehicles.append(vehicle)
    infile.close()
    return my_vehicles


def construct_car_instance(record):
    record = record.strip()
    data_fields = record.split(',')
    vehicle_type, first_name, last_name, street_address_1, street_address_2 = data_fields[0:5]
    city, state, zipcode, make, model, year, color, vehicle_id, fuel_type = data_fields[5:]
    vehicle = Car(first_name,
                  last_name,
                  street_address_1,
                  street_address_2,
                  city,
                  state,
                  zipcode,
                  make,
                  model,
                  int(year),
                  color,
                  vehicle_id,
                  fuel_type)
    return vehicle


def construct_truck_instance(record):
    record = record.strip()
    data_fields = record.split(',')
    vehicle_type, first_name, last_name, street_address_1, street_address_2 = data_fields[0:5]
    city, state, zipcode, make, model, year, color, vehicle_id, gross_weight = data_fields[5:]
    vehicle = Truck(first_name,
                    last_name,
                    street_address_1,
                    street_address_2,
                    city,
                    state,
                    zipcode,
                    make,
                    model,
                    int(year),
                    color,
                    vehicle_id,
                    int(gross_weight))
    return vehicle


def construct_motorcycle_instance(record):
    record = record.strip()
    data_fields = record.split(',')
    vehicle_type, first_name, last_name, street_address_1, street_address_2 = data_fields[0:5]
    city, state, zipcode, make, model, year, color, vehicle_id, displacement_in_ccs = data_fields[5:]
    vehicle = Motorcycle(first_name,
                         last_name,
                         street_address_1,
                         street_address_2,
                         city,
                         state,
                         zipcode,
                         make,
                         model,
                         int(year),
                         color,
                         vehicle_id,
                         int(displacement_in_ccs))
    return vehicle


def by_first_name(vehicle_instance):
    return vehicle_instance.first_name


def by_last_name(vehicle_instance):
    return vehicle_instance.last_name


def by_city(vehicle_instance):
    return vehicle_instance.city

def print_invoices(these_vehicles):
    invoices_printed = 0
    separator_line = f'\n\n{"-" * 45}'
    for vehicle in these_vehicles:
        if isinstance(vehicle, Car):
            subtype = 'Car'
        elif isinstance(vehicle, Truck):
            subtype = "Truck"
        elif isinstance(vehicle, Motorcycle):
            subtype = 'Motorcycle'
        else:
            raise TypeError(f'Unexpected vehicle subtype found.')
        print(separator_line)
        title = f'{subtype} Registration Renewal Invoice'
        print(f'\n{title.upper()}')
        print()
        print(f'{vehicle.first_name} {vehicle.last_name}')
        print(vehicle.street_address_1)
        if vehicle.street_address_2:
            print(vehicle.street_address_2)
        print()
        print(f'{"Make":<20}    {vehicle.make}'),
        print(f'{"Model":<20}   {vehicle.model}'),
        print(f'{"Year":<20}    {vehicle.year}'),
        print(f'{"Year":<20}    {vehicle.color}'),
        print(f'{"Vehicle ID":<20}  {vehicle.vehicle_id}')
        if subtype == 'Car':
            print(f'{"Fuel Type":<20}   {vehicle.fuel_type}')
        elif subtype == 'Truck':
            print(f'{"Gross Weight":<20}    {vehicle.gross_weight:,}')
        elif subtype == 'Motorcycle':
            print(f'{"displacement_in_ccs":<20}     {vehicle.displacement_in_ccs:,}')
        print()

main()
