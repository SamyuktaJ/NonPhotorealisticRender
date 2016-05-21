#pragma once
#include <chrono>
#include <iostream>
#include "windows.h"
#include "psapi.h"

class Performance_Record{
private:
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point end;
	std::chrono::microseconds exe_time;
public:
	__int64 ms;
	Performance_Record();
	~Performance_Record();
	void set_start();
	void set_end();
	void show_memory();
};

Performance_Record::Performance_Record(){
};


Performance_Record::~Performance_Record(){
};
void Performance_Record::set_start(){
	start = std::chrono::steady_clock::now();
};

void Performance_Record::set_end(){
	end = std::chrono::steady_clock::now();
	exe_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	ms = exe_time.count();
	std::cout << (exe_time.count()) / 1000.0 << " ms" << std::endl;
};

void Performance_Record::show_memory()
{
	PROCESS_MEMORY_COUNTERS pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
	printf("INFO: Current memory usage: %f MB \n", double(pmc.WorkingSetSize) / 1024.0 / 1024.0);
};
